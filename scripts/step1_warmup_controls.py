#!/usr/bin/env python3
"""Measure warmed intra-process and HF/HF cached-decode controls for SPEC-01.

This script is diagnostic only. It leaves ADR/status decisions to a human and
uses forced token continuations so post-divergence logits share the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1" / "warmup_controls"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
FORCED_TOKEN_IDS = (198, 220, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)


def _prompt_set() -> dict[str, Any]:
    import yaml

    path = REPO_ROOT / "configs" / "reference_prompts.yaml"
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    data["set_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data


def _render_prompts(tokenizer: Any, config: Any) -> list[dict[str, str]]:
    rendered = []
    for entry in _prompt_set()["prompts"]:
        text = entry["text"] if entry["kind"] == "raw" else tokenizer.apply_chat_template(
            entry["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=config.thinking.enable_thinking,
        )
        rendered.append({"id": entry["id"], "text": text})
    return rendered


def _warm_trace(
    trace_fn: Callable[[], tuple[Any, ...]], policy: Any, label: str
) -> tuple[dict[str, Any], Any]:
    import torch

    for index in range(policy.warmup_traces_per_shape):
        trace_fn()
        print(f"[warmup] {label} warm {index + 1}/{policy.warmup_traces_per_shape}")
    measured = []
    for index in range(policy.measured_traces_per_shape):
        measured.append(trace_fn())
        print(f"[warmup] {label} measure {index + 1}/{policy.measured_traces_per_shape}")
    left, right = measured[-2:]
    if len(left) != len(right):
        raise RuntimeError(f"{label}: measured trace length changed")
    per_step = [bool(torch.equal(a, b)) for a, b in zip(left, right)]
    if policy.require_last_two_exact and not all(per_step):
        raise RuntimeError(f"{label}: final measured traces are not exact")
    return {
        "warmup_traces": policy.warmup_traces_per_shape,
        "measured_traces": policy.measured_traces_per_shape,
        "last_two_exact": all(per_step),
        "exact_steps": sum(per_step),
        "steps": len(per_step),
    }, torch.stack(list(right))


def _warm_sequence(
    sequence_fn: Callable[[], tuple[int, ...]], policy: Any, label: str
) -> tuple[dict[str, Any], tuple[int, ...]]:
    for index in range(policy.warmup_traces_per_shape):
        sequence_fn()
        print(f"[warmup] {label} warm {index + 1}/{policy.warmup_traces_per_shape}")
    measured = []
    for index in range(policy.measured_traces_per_shape):
        measured.append(sequence_fn())
        print(f"[warmup] {label} measure {index + 1}/{policy.measured_traces_per_shape}")
    stable = measured[-2] == measured[-1]
    if policy.require_last_two_exact and not stable:
        raise RuntimeError(f"{label}: final measured sequences are not exact")
    return {
        "warmup_traces": policy.warmup_traces_per_shape,
        "measured_traces": policy.measured_traces_per_shape,
        "last_two_exact": stable,
        "tokens": len(measured[-1]),
    }, measured[-1]


def _compare(left: Any, right: Any) -> dict[str, Any]:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(f"incompatible trace shapes: {tuple(left.shape)} vs {tuple(right.shape)}")
    steps = []
    for index, (actual, reference) in enumerate(zip(left, right)):
        actual = actual.double()
        reference = reference.double()
        delta = torch.abs(actual - reference)
        actual_log_probs = torch.log_softmax(actual, dim=-1)
        reference_log_probs = torch.log_softmax(reference, dim=-1)
        kl = torch.sum(torch.exp(reference_log_probs) * (reference_log_probs - actual_log_probs))
        steps.append(
            {
                "step": index,
                "exact": bool(torch.equal(actual.float(), reference.float())),
                "max_abs_logit_delta": float(delta.max().item()),
                "kl_reference_to_actual_nats": max(0.0, float(kl.item())),
                "top1_actual": int(torch.argmax(actual).item()),
                "top1_reference": int(torch.argmax(reference).item()),
                "top1_agree": bool(torch.argmax(actual) == torch.argmax(reference)),
            }
        )
    return {
        "exact_steps": sum(item["exact"] for item in steps),
        "steps": len(steps),
        "first_divergence": next((item["step"] for item in steps if not item["exact"]), None),
        "max_abs_logit_delta": max(item["max_abs_logit_delta"] for item in steps),
        "max_kl_reference_to_actual_nats": max(item["kl_reference_to_actual_nats"] for item in steps),
        "top1_matches": sum(item["top1_agree"] for item in steps),
        "per_step": steps,
    }


def _native_forced_trace(model: Any, input_ids: Any) -> tuple[Any, ...]:
    import torch

    current = input_ids
    past = None
    trace = []
    with torch.no_grad():
        for token_id in FORCED_TOKEN_IDS:
            outputs = model(input_ids=current, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            trace.append(outputs.logits[0, -1].detach().float().cpu())
            current = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
    return tuple(trace)


def _native_generate(model: Any, tokenizer: Any, config: Any, prompt: str) -> tuple[int, ...]:
    from formic.science.determinism import configure_determinism

    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
    import torch

    device = getattr(model, "device", next(model.parameters()).device)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=len(FORCED_TOKEN_IDS),
        do_sample=False,
        eos_token_id=list(config.generation.eos_token_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return tuple(int(token) for token in output[0, input_ids.shape[-1] :].tolist())


def _write_json(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[artifact] {ARTIFACT_DIR.relative_to(REPO_ROOT) / name}")


def _write_tensors(name: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    import torch

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "tensors": payload}, ARTIFACT_DIR / name)
    print(f"[artifact] {ARTIFACT_DIR.relative_to(REPO_ROOT) / name}")


def _load_direct_hf(config: Any, *, allocation_history_bytes: int = 0) -> tuple[Any, Any, Any]:
    from formic.backbone.torch_compat import ensure_torch_compat
    from formic.science.determinism import configure_determinism

    ensure_torch_compat()
    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    if allocation_history_bytes:
        allocation = torch.empty(allocation_history_bytes // 2, dtype=torch.bfloat16, device="cuda")
        allocation.fill_(0)
        del allocation
        print(f"[control] allocation history: {allocation_history_bytes} bytes")
    raw = json.loads((Path(config.backbone.checkpoint_path) / "config.json").read_text(encoding="utf-8"))
    max_memory = {int(key) if str(key).isdigit() else key: value for key, value in config.backbone.max_memory.items()}
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.backbone.checkpoint_path,
        config=Qwen3_5TextConfig(**raw["text_config"]),
        key_mapping={r"^model\.language_model\.": "model."},
        dtype=torch.bfloat16,
        attn_implementation=config.backbone.attn_implementation,
        device_map=config.backbone.device_map,
        max_memory=max_memory,
    )
    model.eval()
    return model, AutoTokenizer.from_pretrained(config.backbone.checkpoint_path), torch


def stage_intra(config_path: Path) -> None:
    from formic.backbone.loader import load_backbone
    from formic.backbone.runner import forced_cached_decode_logits, generate
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    config = load_config(config_path)
    handle = load_backbone(config)
    device = next(handle.model.parameters()).device
    result: dict[str, Any] = {"stage": "intra", "config_hash": config.config_hash(), "environment": environment_report(), "prompts": {}}
    formic_traces = {}
    native_traces = {}
    for prompt in _render_prompts(handle.tokenizer, config):
        input_ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
        formic_stability, formic_trace = _warm_trace(
            lambda: forced_cached_decode_logits(handle, input_ids[0].tolist(), FORCED_TOKEN_IDS),
            config.numerics,
            f"intra/formic/{prompt['id']}",
        )
        native_stability, native_trace = _warm_trace(
            lambda: _native_forced_trace(handle.model, input_ids),
            config.numerics,
            f"intra/native/{prompt['id']}",
        )
        formic_traces[prompt["id"]] = formic_trace
        native_traces[prompt["id"]] = native_trace
        formic_generate_stability, formic_generated = _warm_sequence(
            lambda: generate(
                handle, prompt["text"], do_sample=False, max_new_tokens=len(FORCED_TOKEN_IDS)
            ).generated_token_ids,
            config.numerics,
            f"intra/formic_generate/{prompt['id']}",
        )
        native_generate_stability, native_generated = _warm_sequence(
            lambda: _native_generate(handle.model, handle.tokenizer, config, prompt["text"]),
            config.numerics,
            f"intra/native_generate/{prompt['id']}",
        )
        result["prompts"][prompt["id"]] = {
            "prompt_length": int(input_ids.shape[-1]),
            "formic_stability": formic_stability,
            "native_stability": native_stability,
            "comparison": _compare(formic_trace, native_trace),
            "generate": {
                "formic_stability": formic_generate_stability,
                "native_stability": native_generate_stability,
                "identical": formic_generated == native_generated,
                "formic_token_ids": list(formic_generated),
                "native_token_ids": list(native_generated),
            },
        }
    _write_json("intra.json", result)
    _write_tensors("intra.pt", {"formic": formic_traces, "native": native_traces}, {"config_hash": config.config_hash(), "trace_kind": "forced_cached_decode"})


def stage_hf(config_path: Path, *, perturbed: bool) -> None:
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    config = load_config(config_path)
    allocation_bytes = 2 * 1024**3 if perturbed else 0
    model, tokenizer, _ = _load_direct_hf(config, allocation_history_bytes=allocation_bytes)
    device = getattr(model, "device", next(model.parameters()).device)
    result: dict[str, Any] = {"stage": "hf_perturbed" if perturbed else "hf_baseline", "config_hash": config.config_hash(), "environment": environment_report(), "allocation_history_bytes": allocation_bytes, "prompts": {}}
    traces = {}
    for prompt in _render_prompts(tokenizer, config):
        input_ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
        stability, trace = _warm_trace(
            lambda: _native_forced_trace(model, input_ids), config.numerics,
            f"{result['stage']}/{prompt['id']}",
        )
        traces[prompt["id"]] = trace
        result["prompts"][prompt["id"]] = {"prompt_length": int(input_ids.shape[-1]), "stability": stability}
    stem = "hf_perturbed" if perturbed else "hf_baseline"
    _write_json(f"{stem}.json", result)
    _write_tensors(f"{stem}.pt", traces, {"config_hash": config.config_hash(), "trace_kind": "forced_cached_decode", "allocation_history_bytes": allocation_bytes})


def stage_hf_compare() -> None:
    import torch

    baseline = torch.load(ARTIFACT_DIR / "hf_baseline.pt", map_location="cpu", weights_only=True)
    perturbed = torch.load(ARTIFACT_DIR / "hf_perturbed.pt", map_location="cpu", weights_only=True)
    if baseline["metadata"]["config_hash"] != perturbed["metadata"]["config_hash"]:
        raise RuntimeError("HF control configs differ")
    result = {"stage": "hf_compare", "config_hash": baseline["metadata"]["config_hash"], "prompts": {}}
    for prompt_id, trace in baseline["tensors"].items():
        result["prompts"][prompt_id] = _compare(trace, perturbed["tensors"][prompt_id])
    _write_json("hf_compare.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 warmed decode controls")
    parser.add_argument("--stage", choices=("intra", "hf-baseline", "hf-perturbed", "hf-compare", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    prepare_backend_environment(load_config(args.config).numerics)
    if args.stage == "intra":
        stage_intra(args.config)
    elif args.stage == "hf-baseline":
        stage_hf(args.config, perturbed=False)
    elif args.stage == "hf-perturbed":
        stage_hf(args.config, perturbed=True)
    elif args.stage == "hf-compare":
        stage_hf_compare()
    else:
        for stage in ("intra", "hf-baseline", "hf-perturbed", "hf-compare"):
            command = [sys.executable, "-u", __file__, "--stage", stage, "--config", str(args.config)]
            result = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
