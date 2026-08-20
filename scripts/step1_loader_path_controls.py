#!/usr/bin/env python3
"""Diagnostic loader-layout and single-load call-path controls for SPEC-01.

The two stages are intentionally independent. ``weights`` loads Formic first,
fingerprints all 851 parameters, releases it, then performs the direct HF load
in the same process. ``paths`` loads Formic once and compares forced cached
traces through the runner helper, stock ``generate()``, and an explicit stock
HF loop. This script records measurements only; it makes no status decision.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1" / "loader_path_controls"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
PROMPT_SET_PATH = REPO_ROOT / "configs" / "reference_prompts.yaml"
EXPECTED_PARAMETERS = 851
FORCED_TOKEN_IDS = (198, 220, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
PATH_NAMES = ("formic_runner", "hf_generate", "hf_explicit")

# The default diagnostic config is fixed. This must be set before importing
# Formic because its package compatibility shim imports torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _write_json(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[artifact] {path.relative_to(REPO_ROOT)}", flush=True)


def _prompt_set() -> dict[str, Any]:
    import yaml

    raw = PROMPT_SET_PATH.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    payload["set_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return payload


def _render_prompts(tokenizer: Any, config: Any) -> list[dict[str, str]]:
    rendered = []
    for entry in _prompt_set()["prompts"]:
        if entry["kind"] == "raw":
            text = entry["text"]
        else:
            text = tokenizer.apply_chat_template(
                entry["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.thinking.enable_thinking,
            )
        rendered.append({"id": entry["id"], "text": text})
    return rendered


def _tensor_sha256(tensor: Any) -> str:
    """Hash logical tensor values as raw bytes without a dtype conversion."""
    import torch

    value = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


def _hooks(hook: Any) -> list[Any]:
    nested = getattr(hook, "hooks", None)
    if nested is None:
        return [hook]
    return [item for child in nested for item in _hooks(child)]


def _resolve_meta_tensor(model: Any, full_name: str) -> Any:
    """Resolve an Accelerate CPU-offloaded parameter from its weights map."""
    import torch

    parts = full_name.split(".")
    for length in range(len(parts) - 1, -1, -1):
        module_name = ".".join(parts[:length])
        relative_name = ".".join(parts[length:])
        module = model.get_submodule(module_name) if module_name else model
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        for candidate in _hooks(hook):
            weights_map = getattr(candidate, "weights_map", None)
            if weights_map is None:
                continue
            try:
                value = weights_map[relative_name]
            except (KeyError, TypeError):
                continue
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(f"cannot resolve offloaded meta tensor {full_name}")


def _memory_format(tensor: Any) -> dict[str, Any]:
    import torch

    contiguous = bool(tensor.is_contiguous(memory_format=torch.contiguous_format))
    channels_last = (
        bool(tensor.is_contiguous(memory_format=torch.channels_last))
        if tensor.ndim == 4
        else None
    )
    channels_last_3d = (
        bool(tensor.is_contiguous(memory_format=torch.channels_last_3d))
        if tensor.ndim == 5
        else None
    )
    if contiguous:
        classification = "contiguous_format"
    elif channels_last:
        classification = "channels_last"
    elif channels_last_3d:
        classification = "channels_last_3d"
    else:
        classification = "non_contiguous"
    return {
        "classification": classification,
        "contiguous_format": contiguous,
        "channels_last": channels_last,
        "channels_last_3d": channels_last_3d,
    }


def _tensor_layout(tensor: Any) -> dict[str, Any]:
    is_meta = tensor.device.type == "meta"
    pointer = None if is_meta else int(tensor.data_ptr())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "memory_format": _memory_format(tensor),
        "data_ptr_mod_256": None if pointer is None else pointer % 256,
        "data_ptr_mod_128": None if pointer is None else pointer % 128,
        "data_ptr_mod_16": None if pointer is None else pointer % 16,
    }


def _model_parameter_fingerprint(model: Any, label: str) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    if len(parameters) != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"{label}: expected {EXPECTED_PARAMETERS} named parameters, got {len(parameters)}"
        )

    entries: dict[str, dict[str, Any]] = {}
    total_numel = 0
    total_bytes = 0
    for index, (name, parameter) in enumerate(parameters, start=1):
        if parameter.device.type == "meta":
            effective = _resolve_meta_tensor(model, name)
            source = "accelerate_weights_map"
        else:
            effective = parameter
            source = "registered_parameter"
        record = {
            "source": source,
            "registered_layout": _tensor_layout(parameter),
            "effective_layout": _tensor_layout(effective),
            "raw_value_sha256": _tensor_sha256(effective),
        }
        entries[name] = record
        total_numel += int(effective.numel())
        total_bytes += int(effective.numel() * effective.element_size())
        if index == 1 or index % 25 == 0 or index == len(parameters):
            print(f"[weights] {label} {index}/{len(parameters)} {name}", flush=True)

    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "parameter_count": len(entries),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "hash_semantics": "raw logical values in contiguous C order; no dtype conversion",
        "parameters": entries,
    }


def _load_direct_hf(config: Any) -> Any:
    from formic.backbone.torch_compat import ensure_torch_compat
    from formic.science.determinism import configure_determinism

    ensure_torch_compat()
    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)

    import torch
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    checkpoint = Path(config.backbone.checkpoint_path)
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    max_memory = {
        int(key) if str(key).isdigit() else key: value
        for key, value in config.backbone.max_memory.items()
    }
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(checkpoint),
        config=Qwen3_5TextConfig(**raw["text_config"]),
        key_mapping={r"^model\.language_model\.": "model."},
        dtype=getattr(torch, config.backbone.dtype),
        attn_implementation=config.backbone.attn_implementation,
        device_map=config.backbone.device_map,
        max_memory=max_memory,
    )
    model.eval()
    return model


def _cuda_memory() -> dict[str, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _finish_release(before: dict[str, int] | None) -> dict[str, Any]:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"before_release": before, "after_release": _cuda_memory()}


def _compare_weight_fingerprints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_entries = left["parameters"]
    right_entries = right["parameters"]
    names = sorted(set(left_entries) | set(right_entries))
    effective_fields = (
        "shape",
        "dtype",
        "device",
        "stride",
        "storage_offset",
        "is_contiguous",
        "memory_format",
        "data_ptr_mod_256",
        "data_ptr_mod_128",
        "data_ptr_mod_16",
    )
    per_tensor: dict[str, dict[str, Any]] = {}
    missing_formic = []
    missing_hf = []
    value_differences = []
    disposition_differences = []
    registered_layout_differences = []

    for name in names:
        formic = left_entries.get(name)
        hf = right_entries.get(name)
        if formic is None:
            missing_formic.append(name)
            per_tensor[name] = {"present_formic": False, "present_hf": True}
            continue
        if hf is None:
            missing_hf.append(name)
            per_tensor[name] = {"present_formic": True, "present_hf": False}
            continue

        value_equal = formic["raw_value_sha256"] == hf["raw_value_sha256"]
        field_equal = {
            field: formic["effective_layout"][field] == hf["effective_layout"][field]
            for field in effective_fields
        }
        registered_equal = formic["registered_layout"] == hf["registered_layout"]
        disposition_equal = all(field_equal.values())
        if not value_equal:
            value_differences.append(name)
        if not disposition_equal:
            disposition_differences.append(name)
        if not registered_equal:
            registered_layout_differences.append(name)
        per_tensor[name] = {
            "present_formic": True,
            "present_hf": True,
            "value_equal": value_equal,
            "effective_disposition_equal": disposition_equal,
            "registered_layout_equal": registered_equal,
            "effective_field_equal": field_equal,
            "formic": formic,
            "hf": hf,
        }

    compared = len(names) - len(missing_formic) - len(missing_hf)
    return {
        "expected_parameters": EXPECTED_PARAMETERS,
        "union_count": len(names),
        "compared_count": compared,
        "missing_formic_count": len(missing_formic),
        "missing_hf_count": len(missing_hf),
        "value_equal_count": compared - len(value_differences),
        "effective_disposition_equal_count": compared - len(disposition_differences),
        "registered_layout_equal_count": compared - len(registered_layout_differences),
        "missing_formic_names": missing_formic,
        "missing_hf_names": missing_hf,
        "value_difference_names": value_differences,
        "effective_disposition_difference_names": disposition_differences,
        "registered_layout_difference_names": registered_layout_differences,
        "per_tensor": per_tensor,
    }


def stage_weights(config_path: Path) -> None:
    from formic.backbone.loader import load_backbone
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    import torch

    config = load_config(config_path)
    run_id = f"{time.time_ns()}-{os.getpid()}"
    started = time.time()

    print("[weights] loading Formic", flush=True)
    handle = load_backbone(config)
    formic_model_report = handle.describe()
    formic = _model_parameter_fingerprint(handle.model, "formic")
    _write_json(
        "weights_formic.json",
        {
            "stage": "weights_formic",
            "run_id": run_id,
            "config_hash": config.config_hash(),
            "environment": environment_report(),
            "model": formic_model_report,
            "fingerprint": formic,
        },
    )

    print("[weights] releasing Formic before direct HF load", flush=True)
    before_release = _cuda_memory()
    handle.boundary_manager.detach()
    del handle
    release = _finish_release(before_release)
    if torch.cuda.is_available() and torch.cuda.memory_allocated() >= 1024**3:
        raise RuntimeError(f"Formic release retained at least 1 GiB on CUDA: {release}")

    print("[weights] loading direct HF", flush=True)
    direct_started = time.time()
    model = _load_direct_hf(config)
    direct_load_seconds = time.time() - direct_started
    direct_model_report = {
        "model_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "named_parameters": len(list(model.named_parameters())),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "load_seconds": direct_load_seconds,
        "memory": _cuda_memory(),
    }
    hf = _model_parameter_fingerprint(model, "hf")
    _write_json(
        "weights_hf.json",
        {
            "stage": "weights_hf",
            "run_id": run_id,
            "config_hash": config.config_hash(),
            "environment": environment_report(),
            "model": direct_model_report,
            "fingerprint": hf,
        },
    )

    comparison = _compare_weight_fingerprints(formic, hf)
    result = {
        "stage": "weights_compare",
        "run_id": run_id,
        "protocol": {
            "process": "single",
            "load_order": ["formic", "direct_hf"],
            "formic_released_before_direct_hf": True,
            "expected_parameters": EXPECTED_PARAMETERS,
            "hash": "SHA-256 of raw logical value bytes without dtype conversion",
        },
        "config_hash": config.config_hash(),
        "environment": environment_report(),
        "release": release,
        "formic_model": formic_model_report,
        "hf_model": direct_model_report,
        "elapsed_seconds": time.time() - started,
        "comparison": comparison,
    }
    _write_json("weights_compare.json", result)
    print(
        "[weights] compared="
        f"{comparison['compared_count']} value_equal={comparison['value_equal_count']} "
        "effective_disposition_equal="
        f"{comparison['effective_disposition_equal_count']}",
        flush=True,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _input_device(model: Any) -> Any:
    import torch

    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    parameter_device = next(model.parameters()).device
    if parameter_device.type == "meta":
        raise RuntimeError("cannot place inputs on the model's meta placeholder device")
    return parameter_device


def _formic_runner_trace(handle: Any, token_ids: list[int]) -> tuple[Any, ...]:
    from formic.backbone.runner import forced_cached_decode_logits

    return forced_cached_decode_logits(handle, token_ids, FORCED_TOKEN_IDS)


def _hf_explicit_trace(model: Any, input_ids: Any) -> tuple[Any, ...]:
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


def _hf_generate_trace(
    model: Any,
    tokenizer: Any,
    config: Any,
    input_ids: Any,
    attention_mask: Any,
) -> tuple[Any, ...]:
    import torch

    prompt_length = int(input_ids.shape[-1])

    def allowed_tokens(_batch_id: int, sequence: Any) -> list[int]:
        step = int(sequence.shape[-1]) - prompt_length
        if not 0 <= step < len(FORCED_TOKEN_IDS):
            raise RuntimeError(f"generate prefix callback received unexpected step {step}")
        return [FORCED_TOKEN_IDS[step]]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=len(FORCED_TOKEN_IDS),
            do_sample=False,
            eos_token_id=list(config.generation.eos_token_ids),
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            prefix_allowed_tokens_fn=allowed_tokens,
            return_dict_in_generate=True,
            output_logits=True,
        )
    generated = tuple(int(token) for token in output.sequences[0, prompt_length:].tolist())
    if generated != FORCED_TOKEN_IDS:
        raise RuntimeError(f"generate did not follow the forced continuation: {generated}")
    if output.logits is None or len(output.logits) != len(FORCED_TOKEN_IDS):
        raise RuntimeError("generate did not return one raw-logit tensor per forced step")
    return tuple(logits[0].detach().float().cpu() for logits in output.logits)


def _trace_stability(left: tuple[Any, ...], right: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    if len(left) != len(right):
        raise RuntimeError(f"measured trace lengths differ: {len(left)} vs {len(right)}")
    per_step = [bool(torch.equal(a, b)) for a, b in zip(left, right)]
    return {
        "last_two_exact": all(per_step),
        "exact_steps": sum(per_step),
        "steps": len(per_step),
        "first_divergence": next((i for i, exact in enumerate(per_step) if not exact), None),
    }


def _trace_metrics(
    actual: Any,
    reference: Any,
    *,
    actual_path: str,
    reference_path: str,
) -> dict[str, Any]:
    import torch

    if tuple(actual.shape) != tuple(reference.shape):
        raise RuntimeError(f"incompatible trace shapes: {actual.shape} vs {reference.shape}")
    per_step = []
    for index, (actual_step, reference_step) in enumerate(zip(actual, reference)):
        actual64 = actual_step.double()
        reference64 = reference_step.double()
        delta = torch.abs(actual64 - reference64)
        actual_log_probs = torch.log_softmax(actual64, dim=-1)
        reference_log_probs = torch.log_softmax(reference64, dim=-1)
        kl = torch.sum(
            torch.exp(reference_log_probs) * (reference_log_probs - actual_log_probs)
        )
        actual_top2 = torch.topk(actual_step, k=2)
        reference_top2 = torch.topk(reference_step, k=2)
        # Match the gate's top-1 definition. torch.topk may choose a different
        # index than argmax when multiple logits are exactly tied.
        actual_top1 = int(torch.argmax(actual_step).item())
        reference_top1 = int(torch.argmax(reference_step).item())
        forced_id = FORCED_TOKEN_IDS[index]
        per_step.append(
            {
                "step": index,
                "forced_token_id": forced_id,
                "exact": bool(torch.equal(actual_step, reference_step)),
                "actual_sha256": _tensor_sha256(actual_step),
                "reference_sha256": _tensor_sha256(reference_step),
                "max_abs_logit_delta": float(delta.max().item()),
                "kl_reference_to_actual_nats": max(0.0, float(kl.item())),
                "actual_top1": actual_top1,
                "reference_top1": reference_top1,
                "top1_agree": actual_top1 == reference_top1,
                "actual_top1_margin": float(
                    (actual_top2.values[0] - actual_top2.values[1]).item()
                ),
                "reference_top1_margin": float(
                    (reference_top2.values[0] - reference_top2.values[1]).item()
                ),
                "actual_forced_logit": float(actual_step[forced_id].item()),
                "reference_forced_logit": float(reference_step[forced_id].item()),
                "actual_forced_log_probability": float(actual_log_probs[forced_id].item()),
                "reference_forced_log_probability": float(
                    reference_log_probs[forced_id].item()
                ),
            }
        )

    return {
        "actual_path": actual_path,
        "reference_path": reference_path,
        "exact_steps": sum(item["exact"] for item in per_step),
        "steps": len(per_step),
        "first_divergence": next(
            (item["step"] for item in per_step if not item["exact"]), None
        ),
        "first_top1_disagreement": next(
            (item["step"] for item in per_step if not item["top1_agree"]), None
        ),
        "max_abs_logit_delta": max(item["max_abs_logit_delta"] for item in per_step),
        "max_kl_reference_to_actual_nats": max(
            item["kl_reference_to_actual_nats"] for item in per_step
        ),
        "top1_matches": sum(item["top1_agree"] for item in per_step),
        "per_step": per_step,
    }


def _measure_prompt_paths(
    handle: Any,
    config: Any,
    prompt: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from formic.science.determinism import configure_determinism

    import torch

    encoded = handle.tokenizer(prompt["text"], return_tensors="pt")
    device = _input_device(handle.model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.shape[0] != 1 or not bool(torch.all(attention_mask == 1)):
        raise RuntimeError("path control requires batch 1 without padding")

    trace_functions: dict[str, Callable[[], tuple[Any, ...]]] = {
        "formic_runner": lambda: _formic_runner_trace(
            handle, [int(token) for token in input_ids[0].tolist()]
        ),
        "hf_generate": lambda: _hf_generate_trace(
            handle.model, handle.tokenizer, config, input_ids, attention_mask
        ),
        "hf_explicit": lambda: _hf_explicit_trace(handle.model, input_ids),
    }
    warmups = config.numerics.warmup_traces_per_shape
    measured_count = config.numerics.measured_traces_per_shape
    measured: dict[str, list[tuple[Any, ...]]] = {name: [] for name in PATH_NAMES}
    execution_order = []

    for cycle in range(warmups + measured_count):
        rotation = cycle % len(PATH_NAMES)
        order = PATH_NAMES[rotation:] + PATH_NAMES[:rotation]
        execution_order.append(list(order))
        phase = "warm" if cycle < warmups else "measure"
        phase_index = cycle + 1 if phase == "warm" else cycle - warmups + 1
        phase_total = warmups if phase == "warm" else measured_count
        for path_name in order:
            configure_determinism(
                config.run.seed, config.run.deterministic, config.numerics
            )
            trace = trace_functions[path_name]()
            if len(trace) != len(FORCED_TOKEN_IDS):
                raise RuntimeError(f"{path_name}: expected 16 logits, got {len(trace)}")
            if phase == "measure":
                measured[path_name].append(trace)
            print(
                f"[paths] {prompt['id']} {path_name} {phase} "
                f"{phase_index}/{phase_total}",
                flush=True,
            )

    stability = {}
    final_traces = {}
    for path_name in PATH_NAMES:
        if len(measured[path_name]) < 2:
            raise RuntimeError(f"{path_name}: at least two measured traces are required")
        report = _trace_stability(measured[path_name][-2], measured[path_name][-1])
        stability[path_name] = {
            "warmup_traces": warmups,
            "measured_traces": measured_count,
            **report,
        }
        if config.numerics.require_last_two_exact and not report["last_two_exact"]:
            raise RuntimeError(
                f"{prompt['id']}/{path_name}: final measured traces are not exact: {report}"
            )
        final_traces[path_name] = torch.stack(list(measured[path_name][-1]))

    comparisons = {
        "formic_runner_vs_hf_generate": _trace_metrics(
            final_traces["formic_runner"],
            final_traces["hf_generate"],
            actual_path="formic_runner",
            reference_path="hf_generate",
        ),
        "formic_runner_vs_hf_explicit": _trace_metrics(
            final_traces["formic_runner"],
            final_traces["hf_explicit"],
            actual_path="formic_runner",
            reference_path="hf_explicit",
        ),
        "hf_generate_vs_hf_explicit": _trace_metrics(
            final_traces["hf_generate"],
            final_traces["hf_explicit"],
            actual_path="hf_generate",
            reference_path="hf_explicit",
        ),
    }
    prompt_result = {
        "prompt_length": int(input_ids.shape[-1]),
        "execution_order": execution_order,
        "stability": stability,
        "comparisons": comparisons,
    }
    return prompt_result, final_traces


def stage_paths(config_path: Path) -> None:
    from formic.backbone.boundaries import count_registered_hooks
    from formic.backbone.loader import load_backbone
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    import torch

    config = load_config(config_path)
    prompt_set = _prompt_set()
    prompt_ids = [entry["id"] for entry in prompt_set["prompts"]]
    if len(prompt_ids) != 6 or len(set(prompt_ids)) != 6:
        raise RuntimeError(f"path control requires six unique prompts, got {prompt_ids}")
    if (
        config.numerics.warmup_traces_per_shape != 6
        or config.numerics.measured_traces_per_shape != 2
        or len(FORCED_TOKEN_IDS) != 16
    ):
        raise RuntimeError(
            "path control requires exactly six warmups, two measured traces, and 16 tokens"
        )
    run_id = f"{time.time_ns()}-{os.getpid()}"
    started = time.time()
    handle = load_backbone(config)
    model_object_id = id(handle.model)
    view_validation = handle.view.validate_against_model(handle.model)
    hook_count = count_registered_hooks(handle.model)
    if hook_count != 0 or not config.identity_mode():
        raise RuntimeError(
            f"single-load path control requires identity mode and zero hooks, got {hook_count}"
        )

    result: dict[str, Any] = {
        "stage": "single_load_paths",
        "run_id": run_id,
        "protocol": {
            "load_count": 1,
            "load_path": "formic.backbone.loader.load_backbone",
            "model_object_id": model_object_id,
            "same_model_object_for_all_paths": True,
            "forced_token_ids": list(FORCED_TOKEN_IDS),
            "tokens_per_trace": len(FORCED_TOKEN_IDS),
            "warmup_traces_per_path_and_prompt": config.numerics.warmup_traces_per_shape,
            "measured_traces_per_path_and_prompt": config.numerics.measured_traces_per_shape,
            "path_definitions": {
                "formic_runner": (
                    "formic.backbone.runner.forced_cached_decode_logits on the BackboneHandle"
                ),
                "hf_generate": (
                    "raw output_logits from handle.model.generate with attention_mask, "
                    "generation-managed position_ids/cache, logits_to_keep=1, and "
                    "prefix-forced tokens"
                ),
                "hf_explicit": (
                    "explicit cached greedy-scoring loop directly on handle.model, with forced "
                    "inputs, position_ids=None, no attention_mask, and default logits_to_keep=0"
                ),
            },
            "group_view_role": (
                "HybridGroupView is validated against the same model; it is a read-only structural "
                "view and has no separate forward implementation"
            ),
        },
        "config_hash": config.config_hash(),
        "seed": config.run.seed,
        "configured_seeds": list(config.run.seeds),
        "prompt_set_sha256": prompt_set["set_sha256"],
        "environment": environment_report(),
        "model": handle.describe(),
        "view_validation": view_validation,
        "registered_layer_hooks": hook_count,
        "prompts": {},
    }
    traces: dict[str, Any] = {}
    for prompt in _render_prompts(handle.tokenizer, config):
        prompt_result, prompt_traces = _measure_prompt_paths(handle, config, prompt)
        result["prompts"][prompt["id"]] = prompt_result
        traces[prompt["id"]] = prompt_traces

    result["elapsed_seconds"] = time.time() - started
    _write_json("paths_compare.json", result)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path = ARTIFACT_DIR / "paths_traces.pt"
    torch.save(
        {
            "metadata": {
                "config_hash": config.config_hash(),
                "seed": config.run.seed,
                "prompt_set_sha256": prompt_set["set_sha256"],
                "forced_token_ids": list(FORCED_TOKEN_IDS),
                "trace_kind": "single_load_forced_cached_decode",
            },
            "tensors": traces,
        },
        tensor_path,
    )
    print(f"[artifact] {tensor_path.relative_to(REPO_ROOT)}", flush=True)

    summary = {}
    for prompt_id, prompt_result in result["prompts"].items():
        summary[prompt_id] = {
            name: {
                key: value
                for key, value in comparison.items()
                if key != "per_step"
            }
            for name, comparison in prompt_result["comparisons"].items()
        }
    print(json.dumps(summary, indent=2), flush=True)

    handle.boundary_manager.detach()
    del handle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_paths_compare(config_path: Path) -> None:
    """Regenerate path metrics from completed traces without loading weights."""
    from formic.config.loader import load_config

    import torch

    config = load_config(config_path)
    json_path = ARTIFACT_DIR / "paths_compare.json"
    tensor_path = ARTIFACT_DIR / "paths_traces.pt"
    if not json_path.is_file() or not tensor_path.is_file():
        raise RuntimeError("paths comparison requires completed JSON and PT artifacts")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if result["config_hash"] != artifact["metadata"]["config_hash"]:
        raise RuntimeError("paths JSON and PT config hashes differ")
    if result["config_hash"] != config.config_hash():
        raise RuntimeError("paths artifacts do not match the requested config")

    pairs = {
        "formic_runner_vs_hf_generate": ("formic_runner", "hf_generate"),
        "formic_runner_vs_hf_explicit": ("formic_runner", "hf_explicit"),
        "hf_generate_vs_hf_explicit": ("hf_generate", "hf_explicit"),
    }
    for prompt_id, prompt_result in result["prompts"].items():
        prompt_traces = artifact["tensors"][prompt_id]
        prompt_result["comparisons"] = {
            name: _trace_metrics(
                prompt_traces[actual],
                prompt_traces[reference],
                actual_path=actual,
                reference_path=reference,
            )
            for name, (actual, reference) in pairs.items()
        }
    result["postprocess"] = {
        "source": "paths_traces.pt",
        "forwards_rerun": False,
        "top1_definition": "torch.argmax",
    }
    _write_json("paths_compare.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 loader and call-path controls")
    parser.add_argument(
        "--stage", choices=("weights", "paths", "paths-compare"), required=True
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    prepare_backend_environment(load_config(args.config).numerics)
    if args.stage == "weights":
        stage_weights(args.config)
    elif args.stage == "paths":
        stage_paths(args.config)
    else:
        stage_paths_compare(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
