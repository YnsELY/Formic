#!/usr/bin/env python3
"""SPEC-01 preliminary verification on the real Qwen3.8-27B checkpoint.

Each weight-bearing stage runs in its own process:

    python scripts/step1_acceptance.py --stage formic
    python scripts/step1_acceptance.py --stage hooks
    python scripts/step1_acceptance.py --stage hf
    python scripts/step1_acceptance.py --stage compare
    python scripts/step1_acceptance.py --stage all

This is deliberately not the formal identity gate. SPEC-02 owns measured
tolerances and blocking CI identity verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
HOOKS_CONFIG = REPO_ROOT / "configs" / "step1_noop_hooks.yaml"

GREEDY_MAX_NEW_TOKENS = 16
SAMPLED_MAX_NEW_TOKENS = 16
SAMPLED_PROMPT_IDS = ("plain_text", "code_completion", "instruction_short")
MANUAL_MAX_NEW_TOKENS = 8
MANUAL_PROMPT_IDS = ("audit_echo", "plain_text", "code_completion", "instruction_short")


def load_prompt_set() -> dict[str, Any]:
    import yaml

    path = REPO_ROOT / "configs" / "reference_prompts_legacy_v1.yaml"
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    data["set_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data


def render_prompts(tokenizer: Any, prompt_set: dict[str, Any], enable_thinking: bool) -> list[dict]:
    rendered = []
    for entry in prompt_set["prompts"]:
        if entry["kind"] == "raw":
            text = entry["text"]
        else:
            text = tokenizer.apply_chat_template(
                entry["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        rendered.append({"id": entry["id"], "kind": entry["kind"], "text": text})
    return rendered


def _cached_greedy_trace(model: Any, input_ids: Any, max_new_tokens: int) -> list[Any]:
    """One fresh-cache trace, retaining logits for the pinned warmup proof."""
    import torch

    current = input_ids
    past = None
    trace = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[0, -1].detach().float().cpu()
            trace.append(logits)
            next_id = int(torch.argmax(logits).item())
            current = torch.tensor([[next_id]], dtype=torch.long, device=input_ids.device)
    return trace


def _warm_cached_decode_shape(
    model: Any, input_ids: Any, config: Any, max_new_tokens: int, *, label: str
) -> tuple[dict[str, Any], Any]:
    """Warm a prompt/cache shape and reject a measurement without stable repeats."""
    import torch

    policy = config.numerics
    for index in range(policy.warmup_traces_per_shape):
        _cached_greedy_trace(model, input_ids, max_new_tokens)
        print(f"[warmup] {label} warm {index + 1}/{policy.warmup_traces_per_shape}")
    measured = []
    for index in range(policy.measured_traces_per_shape):
        measured.append(_cached_greedy_trace(model, input_ids, max_new_tokens))
        print(f"[warmup] {label} measure {index + 1}/{policy.measured_traces_per_shape}")
    left, right = measured[-2:]
    per_step = []
    for step, (actual, reference) in enumerate(zip(left, right)):
        delta = torch.abs(actual.double() - reference.double())
        actual_lp = torch.log_softmax(actual.double(), dim=-1)
        reference_lp = torch.log_softmax(reference.double(), dim=-1)
        kl = torch.sum(torch.exp(reference_lp) * (reference_lp - actual_lp))
        per_step.append(
            {
                "step": step,
                "torch_equal": bool(torch.equal(actual, reference)),
                "max_abs_logit_delta": float(delta.max().item()),
                "kl_nats": max(0.0, float(kl.item())),
                "top1_agree": int(torch.argmax(actual).item()) == int(torch.argmax(reference).item()),
            }
        )
    stable = all(entry["torch_equal"] for entry in per_step)
    result = {
        "prompt_length": int(input_ids.shape[-1]),
        "max_new_tokens": max_new_tokens,
        "warmup_traces": policy.warmup_traces_per_shape,
        "measured_traces": policy.measured_traces_per_shape,
        "last_two_exact": stable,
        "per_step": per_step,
    }
    if policy.require_last_two_exact and not stable:
        raise RuntimeError(f"cached decode warmup did not stabilize: {result}")
    # Persist the final measured trace separately from the JSON summary so a
    # cross-process comparison can retain full per-step logits without bloating it.
    return result, torch.stack(right)


def stage_formic(config_path: Path) -> dict[str, Any]:
    from formic.backbone.boundaries import count_registered_hooks
    from formic.backbone.loader import load_backbone
    from formic.backbone.runner import forward_logits, generate, manual_greedy_decode, set_seed
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    config = load_config(config_path)
    handle = load_backbone(config)
    if count_registered_hooks(handle.model) != 0:
        raise RuntimeError("default Formic config must register zero hooks")

    prompt_set = load_prompt_set()
    prompts = render_prompts(handle.tokenizer, prompt_set, config.thinking.enable_thinking)
    logits: dict[str, Any] = {}
    cached_decode_logits: dict[str, Any] = {}
    results: dict[str, Any] = {
        "stage": "formic",
        "verification_status": "preliminary_spec_01",
        "config_hash": config.config_hash(),
        "prompt_set_sha256": prompt_set["set_sha256"],
        "environment": environment_report(),
        "backbone": handle.describe(),
        "hooks_registered": count_registered_hooks(handle.model),
        "forward": {},
        "cached_decode_stability": {},
        "manual_greedy": {},
        "greedy": {},
        "sampled": {},
    }

    for prompt in prompts:
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
        forward = forward_logits(handle, ids)
        logits[prompt["id"]] = forward.last_logits.float().cpu()
        results["forward"][prompt["id"]] = forward.to_dict()
        print(
            f"[formic] forward {prompt['id']:<20} argmax={forward.argmax_id} "
            f"sha={forward.logits_sha256[:12]} ({forward.seconds:.1f}s)"
        )

    device = next(handle.model.parameters()).device
    for prompt in prompts:
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
        stability, cached_decode_logits[prompt["id"]] = _warm_cached_decode_shape(
            handle.model, ids, config, GREEDY_MAX_NEW_TOKENS, label=f"formic/{prompt['id']}"
        )
        results["cached_decode_stability"][prompt["id"]] = stability

    for prompt in prompts:
        if prompt["id"] not in MANUAL_PROMPT_IDS:
            continue
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
        results["manual_greedy"][prompt["id"]] = manual_greedy_decode(
            handle, ids, MANUAL_MAX_NEW_TOKENS
        )

    for prompt in prompts:
        set_seed(config.run.seed, config.run.deterministic, config.numerics)
        results["greedy"][prompt["id"]] = generate(
            handle,
            prompt["text"],
            do_sample=False,
            max_new_tokens=GREEDY_MAX_NEW_TOKENS,
            seed=config.run.seed,
        ).to_dict()

    for prompt in prompts:
        if prompt["id"] not in SAMPLED_PROMPT_IDS:
            continue
        results["sampled"][prompt["id"]] = generate(
            handle,
            prompt["text"],
            do_sample=True,
            max_new_tokens=SAMPLED_MAX_NEW_TOKENS,
            seed=config.run.seed,
        ).to_dict()

    _write(ARTIFACT_DIR / "formic_outputs.json", results)
    _write_tensors(
        ARTIFACT_DIR / "formic_prefill_logits.pt",
        logits,
        metadata={
            "stage": results["stage"],
            "config_hash": results["config_hash"],
            "prompt_set_sha256": results["prompt_set_sha256"],
            "git_commit": results["environment"]["git_commit"],
        },
    )
    _write_tensors(
        ARTIFACT_DIR / "formic_cached_decode_logits.pt",
        cached_decode_logits,
        metadata={
            "stage": results["stage"],
            "config_hash": results["config_hash"],
            "prompt_set_sha256": results["prompt_set_sha256"],
            "git_commit": results["environment"]["git_commit"],
            "trace_kind": "last_measured_cached_greedy",
        },
    )
    return results


def stage_hooks(config_path: Path, hooks_config_path: Path) -> dict[str, Any]:
    """Compare absent vs 17 registered no-op hooks in one loaded process."""
    import torch

    from formic.backbone import constants as C
    from formic.backbone.boundaries import count_registered_hooks
    from formic.backbone.loader import load_backbone
    from formic.backbone.runner import forward_logits
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    base_config = load_config(config_path)
    hooks_config = load_config(hooks_config_path)
    _assert_hook_configs_compatible(base_config, hooks_config)

    handle = load_backbone(base_config)
    if count_registered_hooks(handle.model) != 0:
        raise RuntimeError("hook proof must start with no registered hooks")

    prompt_set = load_prompt_set()
    prompts = render_prompts(handle.tokenizer, prompt_set, base_config.thinking.enable_thinking)
    absent = {}
    for prompt in prompts:
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
        absent[prompt["id"]] = forward_logits(handle, ids)

    manager = handle.boundary_manager
    manager.configure(
        observers=hooks_config.boundaries.enabled_observers,
        insertions=hooks_config.boundaries.enabled_insertions,
    )
    attached = manager.attach()
    if attached != C.NUM_BOUNDARIES or count_registered_hooks(handle.model) != C.NUM_BOUNDARIES:
        manager.detach()
        raise RuntimeError(f"expected {C.NUM_BOUNDARIES} registered hooks, got {attached}")

    comparisons: dict[str, Any] = {}
    try:
        for prompt in prompts:
            ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
            registered = forward_logits(handle, ids)
            reference = absent[prompt["id"]]
            delta = torch.max(
                torch.abs(reference.last_logits.float() - registered.last_logits.float())
            ).item()
            comparisons[prompt["id"]] = {
                "absent_sha256": reference.logits_sha256,
                "registered_sha256": registered.logits_sha256,
                "same_sha256": reference.logits_sha256 == registered.logits_sha256,
                "torch_equal": bool(torch.equal(reference.last_logits, registered.last_logits)),
                "max_abs_logit_delta": float(delta),
            }
    finally:
        manager.detach()

    results = {
        "stage": "hooks",
        "verification_status": "preliminary_spec_01",
        "base_config_hash": base_config.config_hash(),
        "hooks_config_hash": hooks_config.config_hash(),
        "environment": environment_report(),
        "prompt_set_sha256": prompt_set["set_sha256"],
        "registered_during_forward": attached,
        "registered_after_detach": count_registered_hooks(handle.model),
        "comparisons": comparisons,
        "all_bitwise_identical": all(
            entry["same_sha256"] and entry["torch_equal"]
            for entry in comparisons.values()
        ),
    }
    _write(ARTIFACT_DIR / "hooks_outputs.json", results)
    return results


def stage_hf_text_reference(config_path: Path) -> dict[str, Any]:
    """Direct stock-HF CausalLM reference; no Formic loader or runner code."""
    from formic.backbone.torch_compat import ensure_torch_compat

    ensure_torch_compat()

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    resolved_config = load_config(config_path)
    backbone_cfg = resolved_config.backbone
    if backbone_cfg.mode != "text_only" or backbone_cfg.dtype != "bfloat16":
        raise RuntimeError("SPEC-01 HF reference must be text-only BF16")

    from formic.science.determinism import configure_determinism

    def seed_all(value: int) -> None:
        configure_determinism(value, resolved_config.run.deterministic, resolved_config.numerics)

    seed_all(resolved_config.run.seed)

    checkpoint = Path(backbone_cfg.checkpoint_path)
    raw_checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text_config = Qwen3_5TextConfig(**raw_checkpoint_config["text_config"])
    max_memory = {
        int(key) if str(key).isdigit() else key: value
        for key, value in backbone_cfg.max_memory.items()
    }
    key_mapping = {r"^model\.language_model\.": "model."}

    started = time.time()
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(checkpoint),
        config=text_config,
        key_mapping=key_mapping,
        dtype=torch.bfloat16,
        attn_implementation=backbone_cfg.attn_implementation,
        device_map=backbone_cfg.device_map,
        max_memory=max_memory,
    )
    model.eval()
    load_seconds = time.time() - started
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    device = getattr(model, "device", next(model.parameters()).device)
    if any("visual" in name.split(".") for name, _ in model.named_modules()):
        raise RuntimeError("direct text reference unexpectedly constructed a vision module")

    prompt_set = load_prompt_set()
    prompts = render_prompts(
        tokenizer, prompt_set, resolved_config.thinking.enable_thinking
    )
    logits_artifact: dict[str, Any] = {}
    cached_decode_logits: dict[str, Any] = {}
    results: dict[str, Any] = {
        "stage": "hf_text_reference",
        "verification_status": "preliminary_spec_01",
        "config_hash": resolved_config.config_hash(),
        "environment": environment_report(),
        "model_class": type(model).__name__,
        "key_mapping": key_mapping,
        "prompt_set_sha256": prompt_set["set_sha256"],
        "load_seconds": round(load_seconds, 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "has_vision_tower": False,
        "forward": {},
        "cached_decode_stability": {},
        "manual_greedy": {},
        "greedy": {},
        "sampled": {},
    }

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
            started = time.time()
            output = model(input_ids=ids, use_cache=False)
            seconds = time.time() - started
            logits = output.logits[0, -1].detach().to("cpu").float()
            logits_artifact[prompt["id"]] = logits
            values, indices = torch.topk(logits, k=10)
            results["forward"][prompt["id"]] = {
                "input_token_ids": ids[0].tolist(),
                "argmax_id": int(torch.argmax(logits).item()),
                "top_k_ids": [int(index) for index in indices.tolist()],
                "top_k_values": [float(value) for value in values.tolist()],
                "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                "seconds": seconds,
            }

        for prompt in prompts:
            ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
            stability, cached_decode_logits[prompt["id"]] = _warm_cached_decode_shape(
                model,
                ids,
                resolved_config,
                GREEDY_MAX_NEW_TOKENS,
                label=f"hf/{prompt['id']}",
            )
            results["cached_decode_stability"][prompt["id"]] = stability

        for prompt in prompts:
            if prompt["id"] not in MANUAL_PROMPT_IDS:
                continue
            ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
            past = None
            current = ids
            generated: list[int] = []
            argmax_logits: list[float] = []
            started = time.time()
            for _ in range(MANUAL_MAX_NEW_TOKENS):
                output = model(input_ids=current, past_key_values=past, use_cache=True)
                past = output.past_key_values
                step_logits = output.logits[0, -1].float()
                next_id = int(torch.argmax(step_logits).item())
                generated.append(next_id)
                argmax_logits.append(float(step_logits[next_id].item()))
                current = torch.tensor([[next_id]], dtype=torch.long, device=device)
            results["manual_greedy"][prompt["id"]] = {
                "input_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "argmax_logit_per_step": argmax_logits,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": time.time() - started,
                "cache_seq_length": int(past.get_seq_length()) if past is not None else 0,
            }

        sampling = resolved_config.sampling.payload
        seed = resolved_config.run.seed
        eos_ids = list(resolved_config.generation.eos_token_ids)
        for prompt in prompts:
            encoded = tokenizer(prompt["text"], return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            if ids.shape[0] != 1 or not bool(torch.all(mask == 1)):
                raise RuntimeError("SPEC-01 reference input must be batch 1 without padding")
            seed_all(seed)
            started = time.time()
            output = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=GREEDY_MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            generated = output[0, ids.shape[-1] :].tolist()
            results["greedy"][prompt["id"]] = {
                "prompt_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": time.time() - started,
            }

        for prompt in prompts:
            if prompt["id"] not in SAMPLED_PROMPT_IDS:
                continue
            encoded = tokenizer(prompt["text"], return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            if ids.shape[0] != 1 or not bool(torch.all(mask == 1)):
                raise RuntimeError("SPEC-01 reference input must be batch 1 without padding")
            seed_all(seed)
            started = time.time()
            output = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=SAMPLED_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            generated = output[0, ids.shape[-1] :].tolist()
            results["sampled"][prompt["id"]] = {
                "prompt_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": time.time() - started,
            }

    _write(ARTIFACT_DIR / "hf_outputs.json", results)
    _write_tensors(
        ARTIFACT_DIR / "hf_prefill_logits.pt",
        logits_artifact,
        metadata={
            "stage": results["stage"],
            "config_hash": results["config_hash"],
            "prompt_set_sha256": results["prompt_set_sha256"],
            "git_commit": results["environment"]["git_commit"],
        },
    )
    _write_tensors(
        ARTIFACT_DIR / "hf_cached_decode_logits.pt",
        cached_decode_logits,
        metadata={
            "stage": results["stage"],
            "config_hash": results["config_hash"],
            "prompt_set_sha256": results["prompt_set_sha256"],
            "git_commit": results["environment"]["git_commit"],
            "trace_kind": "last_measured_cached_greedy",
        },
    )
    return results


def stage_compare(config_path: Path, hooks_config_path: Path) -> dict[str, Any]:
    import torch

    from formic.backbone.groups import BOUNDARY_NAMES
    from formic.config.loader import load_config
    from formic.science.determinism import git_commit

    requested_config = load_config(config_path)
    requested_hooks_config = load_config(hooks_config_path)
    formic = _read(ARTIFACT_DIR / "formic_outputs.json")
    hooks = _read(ARTIFACT_DIR / "hooks_outputs.json")
    reference = _read(ARTIFACT_DIR / "hf_outputs.json")
    formic_tensor_artifact = _read_tensors(ARTIFACT_DIR / "formic_prefill_logits.pt")
    reference_tensor_artifact = _read_tensors(ARTIFACT_DIR / "hf_prefill_logits.pt")
    formic_logits = formic_tensor_artifact["tensors"]
    reference_logits = reference_tensor_artifact["tensors"]

    if formic["config_hash"] != requested_config.config_hash():
        raise RuntimeError("Formic artifact config hash does not match --config")
    if reference["config_hash"] != requested_config.config_hash():
        raise RuntimeError("HF artifact config hash does not match --config")
    if hooks["base_config_hash"] != requested_config.config_hash():
        raise RuntimeError("hook artifact base config hash does not match --config")
    if hooks["hooks_config_hash"] != requested_hooks_config.config_hash():
        raise RuntimeError("hook artifact config hash does not match --hooks-config")

    if not (
        formic["prompt_set_sha256"]
        == hooks["prompt_set_sha256"]
        == reference["prompt_set_sha256"]
    ):
        raise RuntimeError("acceptance stages used different prompt sets")

    current_prompt_set = load_prompt_set()
    if current_prompt_set["set_sha256"] != formic["prompt_set_sha256"]:
        raise RuntimeError("acceptance artifacts do not match the current frozen prompt set")
    all_prompt_ids = {entry["id"] for entry in current_prompt_set["prompts"]}
    if not set(MANUAL_PROMPT_IDS) <= all_prompt_ids:
        raise RuntimeError("frozen prompt set is missing a required manual-decode prompt")
    if not set(SAMPLED_PROMPT_IDS) <= all_prompt_ids:
        raise RuntimeError("frozen prompt set is missing a required sampled prompt")
    manual_prompt_ids = all_prompt_ids & set(MANUAL_PROMPT_IDS)
    sampled_prompt_ids = all_prompt_ids & set(SAMPLED_PROMPT_IDS)
    expected_prompt_ids = set(formic["forward"])
    if expected_prompt_ids != all_prompt_ids:
        raise RuntimeError("Formic artifact does not cover every frozen prompt")
    prompt_sets = (
        set(reference["forward"]),
        set(formic_logits),
        set(reference_logits),
        set(hooks["comparisons"]),
    )
    if not expected_prompt_ids or any(prompt_ids != expected_prompt_ids for prompt_ids in prompt_sets):
        raise RuntimeError("acceptance artifacts have missing or extra prompt results")

    for stage_name, payload in (("formic", formic), ("hf", reference)):
        stability = payload.get("cached_decode_stability", {})
        if set(stability) != all_prompt_ids or not all(
            entry["last_two_exact"] for entry in stability.values()
        ):
            raise RuntimeError(f"{stage_name} cached-decode warmup stability proof failed")

    for tensor_artifact, stage in (
        (formic_tensor_artifact, "formic"),
        (reference_tensor_artifact, "hf_text_reference"),
    ):
        metadata = tensor_artifact["metadata"]
        if (
            metadata["stage"] != stage
            or metadata["prompt_set_sha256"] != formic["prompt_set_sha256"]
            or metadata["config_hash"] != requested_config.config_hash()
        ):
            raise RuntimeError(f"{stage} tensor artifact metadata does not match JSON artifact")

    commits = {
        formic["environment"]["git_commit"],
        hooks["environment"]["git_commit"],
        reference["environment"]["git_commit"],
    }
    if len(commits) != 1 or None in commits or git_commit() not in commits:
        raise RuntimeError(f"acceptance artifacts do not share the current commit: {commits}")
    compare_dirty_paths = _git_dirty_paths()
    allowed_registry_paths = {"experiments/REGISTRY.md", "experiments/registry.jsonl"}
    if compare_dirty_paths - allowed_registry_paths:
        raise RuntimeError(
            "compare has uncommitted non-registry changes: "
            f"{sorted(compare_dirty_paths - allowed_registry_paths)}"
        )
    if any(
        tensor_artifact["metadata"]["git_commit"] not in commits
        for tensor_artifact in (formic_tensor_artifact, reference_tensor_artifact)
    ):
        raise RuntimeError("tensor artifacts do not share the JSON artifacts' commit")
    backend_signatures = {
        _backend_signature(payload["environment"])
        for payload in (formic, hooks, reference)
    }
    if len(backend_signatures) != 1:
        raise RuntimeError("acceptance stages used different software or CUDA backends")
    dirty_stages = {
        stage: payload["environment"]["git_dirty"]
        for stage, payload in (("formic", formic), ("hooks", hooks), ("hf", reference))
    }

    forward: dict[str, Any] = {}
    for prompt_id, actual in formic_logits.items():
        expected = reference_logits[prompt_id]
        if actual.ndim != 1 or actual.shape != expected.shape:
            raise RuntimeError(
                f"{prompt_id}: incompatible logit shapes {tuple(actual.shape)} and "
                f"{tuple(expected.shape)}"
            )
        if not bool(torch.all(torch.isfinite(actual))) or not bool(
            torch.all(torch.isfinite(expected))
        ):
            raise RuntimeError(f"{prompt_id}: non-finite logits in comparison artifact")
        delta = torch.abs(actual.double() - expected.double())
        actual_log_probs = torch.log_softmax(actual.double(), dim=-1)
        reference_log_probs = torch.log_softmax(expected.double(), dim=-1)
        kl = torch.sum(
            torch.exp(reference_log_probs) * (reference_log_probs - actual_log_probs)
        )
        formic_entry = formic["forward"][prompt_id]
        reference_entry = reference["forward"][prompt_id]
        actual_f32 = actual.float().contiguous()
        expected_f32 = expected.float().contiguous()
        actual_sha = hashlib.sha256(actual_f32.numpy().tobytes()).hexdigest()
        expected_sha = hashlib.sha256(expected_f32.numpy().tobytes()).hexdigest()
        actual_argmax = int(torch.argmax(actual_f32).item())
        expected_argmax = int(torch.argmax(expected_f32).item())
        if actual_sha != formic_entry["logits_sha256"] or actual_argmax != formic_entry["argmax_id"]:
            raise RuntimeError(f"{prompt_id}: Formic JSON summary does not match tensor artifact")
        if (
            expected_sha != reference_entry["logits_sha256"]
            or expected_argmax != reference_entry["argmax_id"]
        ):
            raise RuntimeError(f"{prompt_id}: HF JSON summary does not match tensor artifact")
        kl_value = float(kl.item())
        if not bool(torch.isfinite(kl)) or kl_value < -1e-10:
            raise RuntimeError(f"{prompt_id}: invalid KL divergence {kl_value}")
        forward[prompt_id] = {
            "same_input_ids": formic_entry["input_token_ids"]
            == reference_entry["input_token_ids"],
            "same_sha256": actual_sha == expected_sha,
            "max_abs_logit_delta": float(delta.max().item()),
            "kl_reference_to_formic_nats_per_token": max(0.0, kl_value),
            "top1_formic": actual_argmax,
            "top1_reference": expected_argmax,
            "top1_agree": actual_argmax == expected_argmax,
        }

    generation = {
        "manual_greedy": _compare_generation(
            formic.get("manual_greedy", {}),
            reference.get("manual_greedy", {}),
            manual_prompt_ids,
        ),
        "greedy": _compare_generation(
            formic.get("greedy", {}), reference.get("greedy", {}), all_prompt_ids
        ),
        "sampled": _compare_generation(
            formic.get("sampled", {}),
            reference.get("sampled", {}),
            sampled_prompt_ids,
        ),
    }
    generation_summary = {
        kind: {
            "exact": sum(entry["identical"] for entry in entries.values()),
            "total": len(entries),
            "minimum_matching_prefix": min(
                (entry["matching_prefix"] for entry in entries.values()), default=0
            ),
        }
        for kind, entries in generation.items()
    }
    max_logit_delta = max(entry["max_abs_logit_delta"] for entry in forward.values())
    kl_values = [entry["kl_reference_to_formic_nats_per_token"] for entry in forward.values()]
    top1_matches = sum(entry["top1_agree"] for entry in forward.values())
    continuous_metrics = {
        "max_abs_logit_delta": max_logit_delta,
        "mean_kl_reference_to_formic_nats_per_token": sum(kl_values) / len(kl_values),
        "max_kl_reference_to_formic_nats_per_token": max(kl_values),
        "top1_agreement": top1_matches / len(forward),
        "top1_matches": top1_matches,
        "evaluated_prompts": len(forward),
    }

    backbone = formic["backbone"]
    mapping = backbone["key_mapping"]
    load_report = backbone["load_report"]
    structure = backbone["structure"]
    forward_exact = all(
        entry["same_sha256"] and entry["same_input_ids"] for entry in forward.values()
    )
    generation_exact = all(
        entry["identical"]
        for kind in ("manual_greedy", "greedy", "sampled")
        for entry in generation[kind].values()
    )
    hooks_exact = bool(hooks["all_bitwise_identical"]) and hooks["registered_after_detach"] == 0
    hooks_cover_all_boundaries = (
        set(requested_hooks_config.boundaries.enabled_insertions) == set(BOUNDARY_NAMES)
        and len(requested_hooks_config.boundaries.enabled_insertions) == len(BOUNDARY_NAMES)
    )
    guard_tests = _run_guard_tests()
    cuda_cumsum_probe = _probe_cuda_cumsum_determinism()

    checklist = [
        {
            "item": "Native greedy and sampled generation matches direct HF CausalLM (preliminary)",
            "pass": forward_exact and generation_exact,
            "detail": (
                f"logit SHA exact on {sum(e['same_sha256'] for e in forward.values())}/"
                f"{len(forward)} prompts; max delta {max_logit_delta:.3e}; "
                f"top-1 {top1_matches}/{len(forward)}; generation exact: "
                f"manual {generation_summary['manual_greedy']['exact']}/"
                f"{generation_summary['manual_greedy']['total']}, greedy "
                f"{generation_summary['greedy']['exact']}/{generation_summary['greedy']['total']}, "
                f"sampled {generation_summary['sampled']['exact']}/"
                f"{generation_summary['sampled']['total']}; CUDA cumsum deterministic="
                f"{cuda_cumsum_probe['supported']}"
            ),
        },
        {
            "item": "Group-to-layer mapping is 16 groups / 64 layers / attention at 3,7,...,63",
            "pass": structure["num_layers"] == 64
            and structure["attention_layer_indices"] == list(range(3, 64, 4)),
            "detail": f"stock mixers {structure['mixer_classes']}",
        },
        {
            "item": "No cell code reimplemented, copied, subclassed, or monkeypatched",
            "pass": guard_tests,
            "detail": "A11 AST/source guard tests",
        },
        {
            "item": "Strict tensor inventory and bijective structural key rename",
            "pass": load_report["ok"]
            and mapping["injective"]
            and mapping["surjective_onto_expected"]
            and mapping["roundtrip"]
            and mapping["metadata_preserved"]
            and mapping["regex_matches_name_map"],
            "detail": (
                f"{load_report['matched']}/851 loaded; mapping "
                f"{mapping['source_tensors']} -> {mapping['target_tensors']}; "
                f"exclusions {load_report['declared_exclusions']}"
            ),
        },
        {
            "item": "Vision tower is not constructed in text-only mode",
            "pass": not load_report["vision_tower_present"]
            and backbone["parameters"] == 26_895_998_464,
            "detail": (
                f"{backbone['parameters']:,} BF16 parameters / "
                f"{backbone['memory']['parameter_bytes'] / 2**30:.2f} GiB; "
                "460,730,096 vision parameters excluded before construction"
            ),
        },
        {
            "item": "All 17 registered no-op boundaries leave real-checkpoint logits unchanged",
            "pass": hooks_exact
            and hooks_cover_all_boundaries
            and hooks["registered_during_forward"] == 17,
            "detail": (
                f"SHA/torch.equal exact on {sum(e['same_sha256'] and e['torch_equal'] for e in hooks['comparisons'].values())}/"
                f"{len(hooks['comparisons'])} prompts in one process"
            ),
        },
        {
            "item": "Every acceptance execution is described by strict YAML config",
            "pass": formic["config_hash"] == hooks["base_config_hash"]
            and reference["config_hash"] == formic["config_hash"]
            and bool(hooks["hooks_config_hash"])
            and not any(dirty_stages.values()),
            "detail": (
                f"default={formic['config_hash'][:12]}, "
                f"noop-hooks={hooks['hooks_config_hash'][:12]}, "
                f"commit={next(iter(commits))[:8]}, dirty={dirty_stages}"
            ),
        },
        {
            "item": "STATUS, experiment registry, ADR template, and conventions exist",
            "pass": all(
                (REPO_ROOT / path).exists()
                for path in (
                    "STATUS.md",
                    "experiments/registry.jsonl",
                    "docs/adr/ADR-TEMPLATE.md",
                    "docs/conventions.md",
                )
            ),
            "detail": "governance files present",
        },
        {
            "item": "SPEC-01 report records A1-A12, measurements, and deviations",
            "pass": _committed_report_is_complete(
                config_hash=formic["config_hash"],
                prompt_hash=formic["prompt_set_sha256"],
                generation_summary=generation_summary,
                continuous_metrics=continuous_metrics,
                hooks=hooks,
                mapping=mapping,
                generation_exact=generation_exact,
            ),
            "detail": "reports/step1_report.md; formal identity gate explicitly deferred to SPEC-02",
        },
    ]

    report = {
        "spec": "SPEC-01",
        "verification_status": "preliminary_only_not_identity_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": formic["config_hash"],
        "hooks_config_hash": hooks["hooks_config_hash"],
        "config_paths": {
            "default": str(config_path.resolve().relative_to(REPO_ROOT)),
            "noop_hooks": str(hooks_config_path.resolve().relative_to(REPO_ROOT)),
        },
        "seeds": [requested_config.run.seed],
        "git_commit": next(iter(commits)),
        "git_dirty_by_stage": dirty_stages,
        "compare_dirty_paths": sorted(compare_dirty_paths),
        "artifact_set_sha256": _artifact_set_sha256(),
        "prompt_set_sha256": formic["prompt_set_sha256"],
        "comparison": {
            "forward": forward,
            "generation": generation,
            "generation_summary": generation_summary,
            "hooks": hooks,
        },
        "decode_diagnostic": {
            "prefill_logits_bitwise_identical": forward_exact,
            "generation_exact": generation_exact,
            "cuda_cumsum_determinism": cuda_cumsum_probe,
            "interpretation": (
                "The stock torch GDN fallback uses cumsum_cuda_kernel while building "
                "cache state. PyTorch reports no deterministic implementation on this "
                "backend. This is a confirmed nondeterministic operation on the decode "
                "path, not by itself a complete causal proof of every token divergence; "
                "SPEC-01 does not patch or replace the stock cell."
            ),
        },
        "continuous_metrics": continuous_metrics,
        "checklist": checklist,
        "preliminary_verification_pass": all(item["pass"] for item in checklist),
    }
    _write(ARTIFACT_DIR / "step1_report.json", report)
    _write_artifact_markdown(report)
    _register_experiment(report, formic, reference)
    _print_checklist(report)
    return report


def _compare_generation(
    actual: dict[str, Any],
    expected: dict[str, Any],
    expected_prompt_ids: set[str],
) -> dict[str, Any]:
    if set(actual) != expected_prompt_ids or set(expected) != expected_prompt_ids:
        raise RuntimeError("generation artifacts have missing or extra prompt results")
    comparison = {}
    for prompt_id, actual_entry in actual.items():
        expected_entry = expected[prompt_id]
        actual_ids = actual_entry["generated_token_ids"]
        expected_ids = expected_entry["generated_token_ids"]
        input_key = "input_token_ids" if "input_token_ids" in actual_entry else "prompt_token_ids"
        same_input_ids = actual_entry[input_key] == expected_entry[input_key]
        prefix = 0
        for left, right in zip(actual_ids, expected_ids):
            if left != right:
                break
            prefix += 1
        comparison[prompt_id] = {
            "identical": same_input_ids and actual_ids == expected_ids,
            "same_input_ids": same_input_ids,
            "matching_prefix": prefix,
            "formic_length": len(actual_ids),
            "reference_length": len(expected_ids),
        }
    return comparison


def _assert_hook_configs_compatible(base: Any, hooks: Any) -> None:
    base_dict = base.to_dict()
    hooks_dict = hooks.to_dict()
    for payload in (base_dict, hooks_dict):
        payload.pop("run")
        payload.pop("boundaries")
    if base_dict != hooks_dict:
        raise RuntimeError("hook proof configs differ in computation settings, not only boundaries/run metadata")
    from formic.backbone.groups import BOUNDARY_NAMES

    if (
        hooks.boundaries.enabled_observers
        or tuple(hooks.boundaries.enabled_insertions) != BOUNDARY_NAMES
    ):
        raise RuntimeError("hook proof config must select exactly 17 no-op insertions")


def _run_guard_tests() -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_no_cell_reimplementation.py",
            "tests/test_inventory.py",
            "tests/test_boundaries.py",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def _probe_cuda_cumsum_determinism() -> dict[str, Any]:
    """Ask PyTorch whether the stock GDN prefill's CUDA cumsum can be deterministic."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "supported": None, "error": None}

    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.ones(8, device="cuda").cumsum(dim=-1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return {"available": True, "supported": False, "error": str(exc)}
    finally:
        torch.use_deterministic_algorithms(was_enabled)
    return {"available": True, "supported": True, "error": None}


def _committed_report_is_complete(
    *,
    config_hash: str,
    prompt_hash: str,
    generation_summary: dict[str, dict[str, int]],
    continuous_metrics: dict[str, Any],
    hooks: dict[str, Any],
    mapping: dict[str, Any],
    generation_exact: bool,
) -> bool:
    path = REPO_ROOT / "reports" / "step1_report.md"
    if not path.is_file():
        return False
    report = path.read_text(encoding="utf-8")
    required = [
        *(f"| A{i} |" for i in range(1, 13)),
        "SPEC-02",
        "## Measurements",
        "## Deviations",
        config_hash,
        prompt_hash,
        f"| Explicit greedy cache loop | {generation_summary['manual_greedy']['exact']}/{generation_summary['manual_greedy']['total']} |",
        f"| Native greedy `generate()` | {generation_summary['greedy']['exact']}/{generation_summary['greedy']['total']} |",
        f"| Native sampled `generate()` | {generation_summary['sampled']['exact']}/{generation_summary['sampled']['total']} |",
        f"| Maximum absolute logit divergence | {continuous_metrics['max_abs_logit_delta']:.6e} |",
        f"| Mean KL(ref || Formic) | {continuous_metrics['mean_kl_reference_to_formic_nats_per_token']:.6e} nats/token |",
        f"| Top-1 agreement | {continuous_metrics['top1_matches']}/{continuous_metrics['evaluated_prompts']} |",
        f"| Registered hooks during proof | {hooks['registered_during_forward']} |",
        f"| Logit SHA-256 equality | {sum(e['same_sha256'] for e in hooks['comparisons'].values())}/{len(hooks['comparisons'])} |",
        f"source tensors       {mapping['source_tensors']}",
        f"target tensors       {mapping['target_tensors']}",
        (
            "**Preliminary SPEC-01 result: PASSED (9/9).**"
            if generation_exact
            else "**Preliminary SPEC-01 result: FAILED (8/9).**"
        ),
    ]
    return all(marker in report for marker in required)


def _artifact_set_sha256() -> str:
    digest = hashlib.sha256()
    for name in (
        "formic_outputs.json",
        "formic_prefill_logits.pt",
        "hooks_outputs.json",
        "hf_outputs.json",
        "hf_prefill_logits.pt",
    ):
        path = ARTIFACT_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_dirty_paths() -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    paths = set()
    for line in result.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def _backend_signature(environment: dict[str, Any]) -> str:
    keys = (
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "cuda_available",
        "cuda_version",
        "gpus",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "flash_sdp",
        "mem_efficient_sdp",
        "math_sdp",
        "deterministic_algorithms",
        "torch_compat",
        "env",
        "has_flash_linear_attention",
        "has_fla",
        "has_causal_conv1d",
        "has_flash_attn",
    )
    return json.dumps({key: environment.get(key) for key in keys}, sort_keys=True)


def _register_experiment(report: dict[str, Any], formic: dict, reference: dict) -> None:
    from formic.science.registry import ExperimentRegistry

    registry = ExperimentRegistry()
    fingerprint_note = f"artifact_set_sha256={report['artifact_set_sha256']}"
    if any(
        fingerprint_note in record.notes
        for record in registry.latest_by_id().values()
    ):
        print(f"[registry] artifact set already registered: {report['artifact_set_sha256'][:12]}")
        return
    record = registry.start(
        title="SPEC-01 preliminary backbone verification",
        step="part1/step1",
        config_hash=report["config_hash"],
        config_path=" + ".join(report["config_paths"].values()),
        seeds=report["seeds"],
        environment=formic.get("environment", {}),
        notes=(
            "Preliminary verification only; formal identity gate belongs to SPEC-02. "
            + fingerprint_note
        ),
    )
    formic_seconds = sum(
        entry["seconds"]
        for section in ("forward", "manual_greedy", "greedy", "sampled")
        for entry in formic[section].values()
    )
    reference_seconds = sum(
        entry["seconds"]
        for section in ("forward", "manual_greedy", "greedy", "sampled")
        for entry in reference[section].values()
    )
    registry.finish(
        record,
        status="DONE" if report["preliminary_verification_pass"] else "FAILED",
        metrics={
            **report["continuous_metrics"],
            "preliminary_verification_pass": report["preliminary_verification_pass"],
            "generation": report["comparison"]["generation_summary"],
            "hooks_bitwise_identical": report["comparison"]["hooks"]["all_bitwise_identical"],
        },
        cost={
            "formic_load_seconds": formic["backbone"]["load_seconds"],
            "reference_load_seconds": reference["load_seconds"],
            "formic_compute_seconds": round(formic_seconds, 1),
            "reference_compute_seconds": round(reference_seconds, 1),
        },
        artifacts=(
            "artifacts/step1/formic_outputs.json",
            "artifacts/step1/hooks_outputs.json",
            "artifacts/step1/hf_outputs.json",
            "artifacts/step1/step1_report.json",
            "artifacts/step1/step1_report.md",
        ),
    )


def _write_artifact_markdown(report: dict[str, Any]) -> None:
    metrics = report["continuous_metrics"]
    lines = [
        "# SPEC-01 preliminary verification",
        "",
        "This is not the formal identity gate. Measured tolerances and blocking CI belong to SPEC-02.",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Config: `{report['config_hash']}`",
        "",
        "## Continuous Formic vs direct-HF metrics",
        "",
        "| Prompt | max abs logit delta | KL(ref || Formic), nats/token | top-1 agree | SHA equal |",
        "|---|---:|---:|---|---|",
    ]
    for prompt_id, entry in report["comparison"]["forward"].items():
        lines.append(
            f"| {prompt_id} | {entry['max_abs_logit_delta']:.6e} | "
            f"{entry['kl_reference_to_formic_nats_per_token']:.6e} | "
            f"{entry['top1_agree']} | {entry['same_sha256']} |"
        )
    lines += [
        "",
        f"Maximum logit divergence: `{metrics['max_abs_logit_delta']:.6e}`",
        f"Mean KL: `{metrics['mean_kl_reference_to_formic_nats_per_token']:.6e}` nats/token",
        f"Top-1 agreement: `{metrics['top1_matches']}/{metrics['evaluated_prompts']}`",
        "",
        "## Decode diagnostic",
        "",
        f"Native generation exact: `{report['decode_diagnostic']['generation_exact']}`.",
        f"CUDA cumsum deterministic implementation available: "
        f"`{report['decode_diagnostic']['cuda_cumsum_determinism']['supported']}`.",
        "",
        report["decode_diagnostic"]["interpretation"],
        "",
        "## Checklist",
        "",
        "| Item | Result | Detail |",
        "|---|---|---|",
    ]
    for item in report["checklist"]:
        lines.append(
            f"| {item['item']} | {'PASS' if item['pass'] else 'FAIL'} | {item['detail']} |"
        )
    lines += [
        "",
        f"**Preliminary verification: {'PASS' if report['preliminary_verification_pass'] else 'FAIL'}**",
        "",
    ]
    (ARTIFACT_DIR / "step1_report.md").write_text("\n".join(lines), encoding="utf-8")


def _print_checklist(report: dict[str, Any]) -> None:
    print("\nSPEC-01 PRELIMINARY VERIFICATION (NOT THE SPEC-02 IDENTITY GATE)")
    for item in report["checklist"]:
        print(f"[{'PASS' if item['pass'] else 'FAIL'}] {item['item']}")
        print(f"       {item['detail']}")
    print(
        f"OVERALL PRELIMINARY RESULT: "
        f"{'PASS' if report['preliminary_verification_pass'] else 'FAIL'}"
    )


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[artifact] {path.relative_to(REPO_ROOT)}")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}; run the corresponding stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tensors(
    path: Path,
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "tensors": payload}, path)
    print(f"[artifact] {path.relative_to(REPO_ROOT)}")


def _read_tensors(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}; run the corresponding stage first")
    return torch.load(path, map_location="cpu", weights_only=True)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 preliminary verification")
    parser.add_argument("--stage", choices=("formic", "hooks", "hf", "compare", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hooks-config", type=Path, default=HOOKS_CONFIG)
    args = parser.parse_args()
    # `--stage all` launches child Python processes. Set cuBLAS before spawning
    # them, before either child imports torch or initializes CUDA.
    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    prepare_backend_environment(load_config(args.config).numerics)

    if args.stage == "all":
        for stage in ("formic", "hooks", "hf", "compare"):
            print(f"\n{'=' * 78}\nSTAGE: {stage}\n{'=' * 78}")
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--stage",
                    stage,
                    "--config",
                    str(args.config),
                    "--hooks-config",
                    str(args.hooks_config),
                ],
                cwd=str(REPO_ROOT),
                env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
            )
            if result.returncode != 0:
                return result.returncode
        return 0

    if args.stage == "formic":
        stage_formic(args.config)
    elif args.stage == "hooks":
        stage_hooks(args.config, args.hooks_config)
    elif args.stage == "hf":
        stage_hf_text_reference(args.config)
    else:
        report = stage_compare(args.config, args.hooks_config)
        return 0 if report["preliminary_verification_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
