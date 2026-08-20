#!/usr/bin/env python3
"""Read-only call and cache-state diagnostics for the SPEC-01 runner.

The observer gate must pass before state capture is attempted. Observers are
registered only on the top-level CausalLM module, return ``None``, and are
removed after each trace. No Qwen cell or Formic boundary is hooked.

Cache handling follows A1-A4: caches are created by the stock model, only used
forward, never cropped/restored/forked, and state capture retains no device
clone. Raw bytes are copied synchronously to CPU before the next in-place cache
update.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1" / "runner_state_diagnostics"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
PROMPT_SET_PATH = REPO_ROOT / "configs" / "reference_prompts.yaml"
FORCED_TOKEN_IDS = (198, 220, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
PATH_NAMES = ("formic_runner", "hf_generate", "hf_explicit")
STATE_PATH_NAMES = ("formic_runner", "hf_explicit")
OBSERVED_ARGUMENTS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "cache_position",
    "logits_to_keep",
    "num_logits_to_keep",
    "use_cache",
    "past_key_values",
)

# The default diagnostic config is fixed. Pin cuBLAS before importing Formic,
# whose package compatibility shim imports torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _tensor_sha256(tensor: Any) -> str:
    """Hash raw logical tensor bytes without changing dtype or device state."""
    value = tensor.detach().to("cpu").contiguous().reshape(-1).view(_torch().uint8)
    return _sha256_bytes(memoryview(value.numpy()))


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "contiguous": bool(tensor.is_contiguous()),
        "sha256": _tensor_sha256(tensor),
    }


def _default_record(model: Any, name: str, value: Any) -> dict[str, Any]:
    if value is inspect.Parameter.empty:
        return {"available": False}
    if name == "use_cache" and value is None and hasattr(model.config, "use_cache"):
        return {
            "available": True,
            "source": "model.config.use_cache",
            "value": _simple_value_record(model.config.use_cache),
        }
    return {
        "available": True,
        "source": "forward_signature",
        "value": _simple_value_record(value),
    }


def _simple_value_record(value: Any) -> dict[str, Any]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return _tensor_record(value)
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "items": [_simple_value_record(item) for item in value],
        }
    return {"kind": "object", "type": _qualified_type(value), "repr": repr(value)}


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _cache_call_record(cache: Any) -> dict[str, Any]:
    if cache is None:
        return {"kind": "none"}
    layers = list(getattr(cache, "layers", ()))
    try:
        sequence_length = int(cache.get_seq_length())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        sequence_length = None
    layer_records = []
    for index, layer in enumerate(layers):
        try:
            layer_length = int(layer.get_seq_length())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            layer_length = None
        layer_records.append(
            {
                "layer": index,
                "type": _qualified_type(layer),
                "sequence_length": layer_length,
                "is_initialized": bool(getattr(layer, "is_initialized", False)),
                "is_conv_states_initialized": bool(
                    getattr(layer, "is_conv_states_initialized", False)
                ),
                "is_recurrent_states_initialized": bool(
                    getattr(layer, "is_recurrent_states_initialized", False)
                ),
                "has_previous_state": bool(getattr(layer, "has_previous_state", False)),
            }
        )
    return {
        "kind": "cache",
        "type": _qualified_type(cache),
        "sequence_length": sequence_length,
        "layer_count": len(layers),
        "layers": layer_records,
    }


def _forward_defaults(model: Any) -> dict[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    return {
        name: parameters[name].default if name in parameters else inspect.Parameter.empty
        for name in OBSERVED_ARGUMENTS
    }


def _argument_record(
    model: Any,
    name: str,
    supplied: dict[str, Any],
    sources: dict[str, str],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if name not in supplied:
        return {
            "status": "absent",
            "effective_default": _default_record(
                model, name, defaults.get(name, inspect.Parameter.empty)
            ),
        }
    value = supplied[name]
    record = _cache_call_record(value) if name == "past_key_values" else _simple_value_record(value)
    return {"status": "present", "source": sources[name], "value": record}


def _supplied_arguments(
    model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    signature = inspect.signature(model.forward)
    bound = signature.bind_partial(*args, **kwargs)
    supplied: dict[str, Any] = {}
    sources: dict[str, str] = {}
    positional_names = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    positional = set(positional_names[: len(args)])
    for name, value in bound.arguments.items():
        parameter = signature.parameters[name]
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            supplied.update(value)
            sources.update({key: "keyword" for key in value})
        else:
            supplied[name] = value
            sources[name] = "positional" if name in positional else "keyword"
    return supplied, sources


def _call_record(
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    call_index: int,
) -> dict[str, Any]:
    defaults = _forward_defaults(model)
    supplied, sources = _supplied_arguments(model, args, kwargs)
    tracked = {
        name: _argument_record(model, name, supplied, sources, defaults)
        for name in OBSERVED_ARGUMENTS
    }
    additional = {
        name: _simple_value_record(value)
        for name, value in sorted(supplied.items())
        if name not in OBSERVED_ARGUMENTS
    }
    return {
        "call_index": call_index,
        "boundary": _boundary_label(call_index),
        "legacy_logit_step": call_index,
        "arguments": tracked,
        "additional_arguments": additional,
    }


def _boundary_label(call_index: int) -> str:
    return "prefill" if call_index == 0 else f"after_forced_{call_index - 1}"


class _TopLevelObserver:
    """Read-only hooks on one top-level CausalLM object."""

    def __init__(self, *, capture_state: bool):
        self.capture_state = capture_state
        self.calls: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []

    def pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append(_call_record(module, args, kwargs, len(self.calls)))
        return None

    def post_hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if self.capture_state:
            call_index = len(self.states)
            self.states.append(
                _state_snapshot(module, output.past_key_values, call_index)
            )
        return None


@contextmanager
def _observe_top_level(
    model: Any, *, capture_state: bool
) -> Iterator[_TopLevelObserver]:
    observer = _TopLevelObserver(capture_state=capture_state)
    pre_handle = None
    post_handle = None
    try:
        pre_handle = model.register_forward_pre_hook(observer.pre_hook, with_kwargs=True)
        if capture_state:
            post_handle = model.register_forward_hook(observer.post_hook, with_kwargs=True)
        yield observer
    finally:
        if pre_handle is not None:
            pre_handle.remove()
        if post_handle is not None:
            post_handle.remove()


def _component_record(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "none"}
    record = _tensor_record(value)
    return {"status": "tensor", **record}


def _rope_records(model: Any) -> list[dict[str, Any]]:
    records = []
    for module_name, module in model.named_modules():
        if "rope_deltas" not in vars(module):
            continue
        value = vars(module)["rope_deltas"]
        records.append(
            {
                "module": module_name or "<root>",
                "value": _component_record(value),
            }
        )
    return records or [{"module": None, "value": {"status": "absent"}}]


def _state_snapshot(model: Any, cache: Any, call_index: int) -> dict[str, Any]:
    layers = []
    for index, layer in enumerate(getattr(cache, "layers", ())):
        if hasattr(layer, "keys") or hasattr(layer, "values"):
            components = {
                "K": _component_record(getattr(layer, "keys", None)),
                "V": _component_record(getattr(layer, "values", None)),
            }
            kind = "full_attention"
        else:
            components = {
                "conv_states": _component_record(getattr(layer, "conv_states", None)),
                "recurrent_states": _component_record(
                    getattr(layer, "recurrent_states", None)
                ),
            }
            kind = "gdn"
        try:
            sequence_length = int(layer.get_seq_length())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            sequence_length = None
        layers.append(
            {
                "layer": index,
                "kind": kind,
                "type": _qualified_type(layer),
                "sequence_length": sequence_length,
                "components": components,
                "aggregate_sha256": _json_sha256(components),
                "is_initialized": bool(getattr(layer, "is_initialized", False)),
                "is_conv_states_initialized": bool(
                    getattr(layer, "is_conv_states_initialized", False)
                ),
                "is_recurrent_states_initialized": bool(
                    getattr(layer, "is_recurrent_states_initialized", False)
                ),
                "has_previous_state": bool(getattr(layer, "has_previous_state", False)),
            }
        )
    ropes = _rope_records(model)
    return {
        "call_index": call_index,
        "boundary": _boundary_label(call_index),
        "legacy_logit_step": call_index,
        "cache_type": _qualified_type(cache),
        "layers": layers,
        "rope_deltas": ropes,
        "aggregate_sha256": _json_sha256({"layers": layers, "rope_deltas": ropes}),
    }


def _validate_state_trace(
    states: list[dict[str, Any]], prompt_length: int
) -> dict[str, Any]:
    expected_attention = set(range(3, 64, 4))
    if len(states) != len(FORCED_TOKEN_IDS):
        raise RuntimeError(
            f"expected {len(FORCED_TOKEN_IDS)} state boundaries, got {len(states)}"
        )
    for call_index, state in enumerate(states):
        if state["call_index"] != call_index or state["boundary"] != _boundary_label(call_index):
            raise RuntimeError(f"state boundary ordering changed at call {call_index}")
        if len(state["layers"]) != 64:
            raise RuntimeError(
                f"{state['boundary']}: expected 64 cache layers, got {len(state['layers'])}"
            )
        attention = {
            layer["layer"] for layer in state["layers"] if layer["kind"] == "full_attention"
        }
        gdn = {layer["layer"] for layer in state["layers"] if layer["kind"] == "gdn"}
        if attention != expected_attention or gdn != set(range(64)) - expected_attention:
            raise RuntimeError(
                f"{state['boundary']}: cache layer layout differs from 48 GDN / 16 full attention"
            )
        expected_length = prompt_length + call_index
        for layer in state["layers"]:
            if layer["kind"] == "full_attention":
                if layer["sequence_length"] != expected_length:
                    raise RuntimeError(
                        f"{state['boundary']}/layer {layer['layer']}: expected KV length "
                        f"{expected_length}, got {layer['sequence_length']}"
                    )
                if not layer["is_initialized"]:
                    raise RuntimeError(
                        f"{state['boundary']}/layer {layer['layer']}: attention cache is uninitialized"
                    )
                required = ("K", "V")
            else:
                if not (
                    layer["is_conv_states_initialized"]
                    and layer["is_recurrent_states_initialized"]
                    and layer["has_previous_state"]
                ):
                    raise RuntimeError(
                        f"{state['boundary']}/layer {layer['layer']}: GDN state is incomplete"
                    )
                required = ("conv_states", "recurrent_states")
            for component in required:
                if layer["components"][component]["status"] != "tensor":
                    raise RuntimeError(
                        f"{state['boundary']}/layer {layer['layer']}/{component}: tensor is missing"
                    )
    return {
        "valid": True,
        "boundaries": len(states),
        "layers_per_boundary": 64,
        "full_attention_layers": sorted(expected_attention),
        "gdn_layers": sorted(set(range(64)) - expected_attention),
    }


def _torch() -> Any:
    import torch

    return torch


def _prompt_set() -> dict[str, Any]:
    import yaml

    raw = PROMPT_SET_PATH.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    payload["set_sha256"] = _sha256_bytes(raw.encode("utf-8"))
    return payload


def _render_prompts(tokenizer: Any, config: Any) -> list[dict[str, str]]:
    rendered = []
    for entry in _prompt_set()["prompts"]:
        text = entry["text"] if entry["kind"] == "raw" else tokenizer.apply_chat_template(
            entry["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.thinking.enable_thinking,
        )
        rendered.append({"id": entry["id"], "text": text})
    return rendered


def _input_device(model: Any) -> Any:
    torch = _torch()
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    parameter_device = next(model.parameters()).device
    if parameter_device.type == "meta":
        raise RuntimeError("cannot place input on a meta placeholder device")
    return parameter_device


def _runner_trace(handle: Any, token_ids: list[int]) -> tuple[Any, ...]:
    from formic.backbone.runner import forced_cached_decode_logits

    return forced_cached_decode_logits(handle, token_ids, FORCED_TOKEN_IDS)


def _explicit_trace(model: Any, input_ids: Any) -> tuple[Any, ...]:
    torch = _torch()
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


def _generate_trace(
    model: Any,
    tokenizer: Any,
    config: Any,
    input_ids: Any,
    attention_mask: Any,
) -> tuple[Any, ...]:
    torch = _torch()
    prompt_length = int(input_ids.shape[-1])

    def allowed_tokens(_batch_id: int, sequence: Any) -> list[int]:
        step = int(sequence.shape[-1]) - prompt_length
        if not 0 <= step < len(FORCED_TOKEN_IDS):
            raise RuntimeError(f"generate callback received unexpected step {step}")
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
        raise RuntimeError(f"generate did not follow forced continuation: {generated}")
    if output.logits is None or len(output.logits) != len(FORCED_TOKEN_IDS):
        raise RuntimeError("generate did not return one logit tensor per forced step")
    return tuple(logits[0].detach().float().cpu() for logits in output.logits)


def _trace_function(
    path_name: str,
    handle: Any,
    input_ids: Any,
    attention_mask: Any,
) -> Callable[[], tuple[Any, ...]]:
    if path_name == "formic_runner":
        token_ids = [int(token) for token in input_ids[0].tolist()]
        return lambda: _runner_trace(handle, token_ids)
    if path_name == "hf_explicit":
        return lambda: _explicit_trace(handle.model, input_ids)
    if path_name == "hf_generate":
        return lambda: _generate_trace(
            handle.model, handle.tokenizer, handle.config, input_ids, attention_mask
        )
    raise ValueError(f"unknown path {path_name!r}")


def _run_trace(
    trace_fn: Callable[[], tuple[Any, ...]],
    model: Any,
    *,
    observed: bool,
    capture_state: bool,
) -> tuple[tuple[Any, ...], _TopLevelObserver | None]:
    if not observed:
        return trace_fn(), None
    with _observe_top_level(model, capture_state=capture_state) as observer:
        trace = trace_fn()
    if len(observer.calls) != len(trace):
        raise RuntimeError(
            f"observer recorded {len(observer.calls)} calls for {len(trace)} logits"
        )
    if capture_state and len(observer.states) != len(trace):
        raise RuntimeError(
            f"observer recorded {len(observer.states)} states for {len(trace)} logits"
        )
    return trace, observer


def _trace_metrics(left: tuple[Any, ...], right: tuple[Any, ...]) -> dict[str, Any]:
    torch = _torch()
    if len(left) != len(right):
        raise RuntimeError(f"trace lengths differ: {len(left)} vs {len(right)}")
    steps = []
    for index, (actual, reference) in enumerate(zip(left, right)):
        delta = torch.abs(actual.double() - reference.double())
        steps.append(
            {
                "step": index,
                "boundary": _boundary_label(index),
                "exact": bool(torch.equal(actual, reference)),
                "actual_sha256": _tensor_sha256(actual),
                "reference_sha256": _tensor_sha256(reference),
                "max_abs_logit_delta": float(delta.max().item()),
                "actual_top1": int(torch.argmax(actual).item()),
                "reference_top1": int(torch.argmax(reference).item()),
                "top1_agree": bool(torch.argmax(actual) == torch.argmax(reference)),
            }
        )
    return {
        "exact_steps": sum(step["exact"] for step in steps),
        "steps": len(steps),
        "first_divergence": next((step["step"] for step in steps if not step["exact"]), None),
        "first_divergence_boundary": next(
            (step["boundary"] for step in steps if not step["exact"]), None
        ),
        "max_abs_logit_delta": max(step["max_abs_logit_delta"] for step in steps),
        "top1_matches": sum(step["top1_agree"] for step in steps),
        "per_step": steps,
    }


def _diff_values(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                differences.append({"field": path, "left": {"status": "missing"}, "right": right[key]})
            elif key not in right:
                differences.append({"field": path, "left": left[key], "right": {"status": "missing"}})
            else:
                differences.extend(_diff_values(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left):
                differences.append({"field": path, "left": {"status": "missing"}, "right": right[index]})
            elif index >= len(right):
                differences.append({"field": path, "left": left[index], "right": {"status": "missing"}})
            else:
                differences.extend(_diff_values(left[index], right[index], path))
        return differences
    if left != right:
        differences.append({"field": prefix, "left": left, "right": right})
    return differences


def _call_log_diff(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    differences = _diff_values(left, right, "calls")
    return {
        "exact": not differences,
        "difference_count": len(differences),
        "first_difference": differences[0] if differences else None,
        "differences": differences,
    }


def _state_component_diff(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    differences = []
    for boundary_index in range(min(len(left), len(right))):
        left_state = left[boundary_index]
        right_state = right[boundary_index]
        if left_state["cache_type"] != right_state["cache_type"]:
            differences.append(
                {
                    "boundary_index": boundary_index,
                    "boundary": left_state["boundary"],
                    "layer": None,
                    "layer_kind": "cache",
                    "component": "cache_type",
                    "left": left_state["cache_type"],
                    "right": right_state["cache_type"],
                }
            )
        if len(left_state["layers"]) != len(right_state["layers"]):
            differences.append(
                {
                    "boundary_index": boundary_index,
                    "boundary": left_state["boundary"],
                    "layer": None,
                    "layer_kind": "cache",
                    "component": "layer_count",
                    "left": len(left_state["layers"]),
                    "right": len(right_state["layers"]),
                }
            )
        for layer_index in range(min(len(left_state["layers"]), len(right_state["layers"]))):
            left_layer = left_state["layers"][layer_index]
            right_layer = right_state["layers"][layer_index]
            for field in (
                "kind",
                "type",
                "sequence_length",
                "is_initialized",
                "is_conv_states_initialized",
                "is_recurrent_states_initialized",
                "has_previous_state",
            ):
                if left_layer[field] != right_layer[field]:
                    differences.append(
                        {
                            "boundary_index": boundary_index,
                            "boundary": left_state["boundary"],
                            "layer": layer_index,
                            "layer_kind": left_layer["kind"],
                            "component": f"metadata.{field}",
                            "left": left_layer[field],
                            "right": right_layer[field],
                        }
                    )
            component_names = sorted(
                set(left_layer["components"]) | set(right_layer["components"])
            )
            for component in component_names:
                left_component = left_layer["components"].get(component, {"status": "missing"})
                right_component = right_layer["components"].get(component, {"status": "missing"})
                if left_component != right_component:
                    differences.append(
                        {
                            "boundary_index": boundary_index,
                            "boundary": left_state["boundary"],
                            "layer": layer_index,
                            "layer_kind": left_layer["kind"],
                            "component": component,
                            "left": left_component,
                            "right": right_component,
                        }
                    )
        if left_state["rope_deltas"] != right_state["rope_deltas"]:
            differences.append(
                {
                    "boundary_index": boundary_index,
                    "boundary": left_state["boundary"],
                    "layer": None,
                    "layer_kind": "model_state",
                    "component": "rope_deltas",
                    "left": left_state["rope_deltas"],
                    "right": right_state["rope_deltas"],
                }
            )
    if len(left) != len(right):
        differences.append(
            {
                "boundary_index": min(len(left), len(right)),
                "boundary": None,
                "layer": None,
                "layer_kind": None,
                "component": "trace_length",
                "left": len(left),
                "right": len(right),
            }
        )
    return {
        "exact": not differences,
        "difference_count": len(differences),
        "first_difference": differences[0] if differences else None,
        "differences": differences,
    }


def _pip_freeze() -> dict[str, Any]:
    result = subprocess.run(
        (sys.executable, "-m", "pip", "freeze"),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"pip freeze failed: {result.stderr.strip()}")
    return {
        "command": f"{sys.executable} -m pip freeze",
        "stdout": result.stdout,
        "sha256": _sha256_bytes(result.stdout.encode("utf-8")),
    }


def _runtime_source_manifest(config: Any) -> dict[str, Any]:
    import accelerate
    import transformers

    files: set[Path] = {
        Path(__file__).resolve(),
        DEFAULT_CONFIG.resolve(),
        PROMPT_SET_PATH.resolve(),
    }
    files.update((REPO_ROOT / "formic").glob("**/*.py"))
    files.update(Path(transformers.__file__).resolve().parent.glob("**/*.py"))
    files.update(Path(accelerate.__file__).resolve().parent.glob("**/*.py"))
    checkpoint = Path(config.backbone.checkpoint_path)
    files.update(checkpoint.glob("*.json"))
    files.update(checkpoint.glob("*.jinja"))
    content_entries = {
        str(path): {
            "size": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
        }
        for path in sorted(files)
    }
    shard_entries = {}
    metadata_dir = checkpoint / ".cache" / "huggingface" / "download"
    for path in sorted(checkpoint.glob("*.safetensors")):
        metadata_path = metadata_dir / f"{path.name}.metadata"
        metadata_lines = (
            metadata_path.read_text(encoding="utf-8").splitlines()
            if metadata_path.is_file()
            else []
        )
        shard_entries[path.name] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "huggingface_commit": metadata_lines[0] if len(metadata_lines) >= 1 else None,
            "lfs_sha256": metadata_lines[1] if len(metadata_lines) >= 2 else None,
            "metadata_sha256": (
                _sha256_bytes(metadata_path.read_bytes()) if metadata_path.is_file() else None
            ),
        }
    audit_integrity = Path("/workspace/audits/qwen3_8_27b/results/integrity_provenance.json")
    return {
        "content_files": content_entries,
        "safetensors_local_identity": shard_entries,
        "audit_integrity_manifest": (
            {
                "path": str(audit_integrity),
                "sha256": _sha256_bytes(audit_integrity.read_bytes()),
            }
            if audit_integrity.is_file()
            else None
        ),
    }


def _metadata(config: Any, handle: Any, stage: str) -> dict[str, Any]:
    from formic.science.determinism import environment_report, git_commit, git_dirty

    environment = environment_report()
    pip_freeze = _pip_freeze()
    model = handle.describe()
    source_manifest = _runtime_source_manifest(config)
    protocol_identity = {
        "config_hash": config.config_hash(),
        "prompt_set_sha256": _prompt_set()["set_sha256"],
        "forced_token_ids": list(FORCED_TOKEN_IDS),
        "runtime_source_manifest_sha256": _json_sha256(source_manifest),
        "model": {
            "model_class": model["model_class"],
            "parameters": model["parameters"],
            "dtype": model["dtype"],
            "attn_implementation": model["attn_implementation"],
        },
        "environment": {
            key: environment.get(key)
            for key in (
                "torch",
                "cuda_version",
                "gpus",
                "transformers",
                "accelerate",
                "safetensors",
                "cudnn_deterministic",
                "cudnn_benchmark",
                "cudnn_allow_tf32",
                "cuda_matmul_allow_tf32",
                "flash_sdp",
                "mem_efficient_sdp",
                "math_sdp",
            )
        },
        "pip_freeze_sha256": pip_freeze["sha256"],
    }
    return {
        "stage": stage,
        "created_unix_ns": time.time_ns(),
        "config_hash": config.config_hash(),
        "prompt_set_sha256": _prompt_set()["set_sha256"],
        "forced_token_ids": list(FORCED_TOKEN_IDS),
        "git_commit": git_commit(REPO_ROOT),
        "git_dirty": git_dirty(REPO_ROOT),
        "environment": environment,
        "pip_freeze": pip_freeze,
        "model": model,
        "runtime_source_manifest": source_manifest,
        "protocol_identity": protocol_identity,
        "protocol_identity_sha256": _json_sha256(protocol_identity),
    }


def _write_json(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[artifact] {path.relative_to(REPO_ROOT)}", flush=True)


def _write_tensors(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    _torch().save(payload, path)
    print(f"[artifact] {path.relative_to(REPO_ROOT)}", flush=True)


def _load_handle(config_path: Path) -> tuple[Any, Any]:
    from formic.backbone.boundaries import count_registered_hooks
    from formic.backbone.loader import load_backbone
    from formic.config.loader import load_config

    config = load_config(config_path)
    handle = load_backbone(config)
    hooks = count_registered_hooks(handle.model)
    if hooks or not config.identity_mode():
        raise RuntimeError(
            f"diagnostics require identity mode and zero boundary hooks, got {hooks}"
        )
    return config, handle


def _validate_protocol(config: Any) -> None:
    prompt_ids = [prompt["id"] for prompt in _prompt_set()["prompts"]]
    if len(prompt_ids) != 6 or len(set(prompt_ids)) != 6:
        raise RuntimeError(f"expected six unique frozen prompts, got {prompt_ids}")
    if config.numerics.warmup_traces_per_shape != 6:
        raise RuntimeError("diagnostic requires six configured warmup traces")
    if config.numerics.measured_traces_per_shape != 2:
        raise RuntimeError("diagnostic requires two configured measured traces")
    if not config.numerics.require_last_two_exact:
        raise RuntimeError("diagnostic requires exact final-trace stability")


def stage_observer_gate(config_path: Path) -> None:
    """Blocking bit-inertness gate for the top-level argument observer."""
    from formic.science.determinism import configure_determinism

    torch = _torch()
    config, handle = _load_handle(config_path)
    _validate_protocol(config)
    result = {
        "metadata": _metadata(config, handle, "observer_gate"),
        "protocol": {
            "paths": list(PATH_NAMES),
            "modes": ["naked", "observed"],
            "warmups_per_path_mode_prompt": config.numerics.warmup_traces_per_shape,
            "measured_per_path_mode_prompt": config.numerics.measured_traces_per_shape,
            "observer_scope": "top-level CausalLM only",
            "observer_returns": None,
            "path_order_rotates": True,
            "mode_order_alternates": True,
        },
        "prompts": {},
    }
    tensor_payload = {}
    failures = []
    for prompt in _render_prompts(handle.tokenizer, config):
        encoded = handle.tokenizer(prompt["text"], return_tensors="pt")
        device = _input_device(handle.model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[0] != 1 or not bool(torch.all(attention_mask == 1)):
            raise RuntimeError("observer gate requires batch 1 without padding")
        trace_functions = {
            name: _trace_function(name, handle, input_ids, attention_mask)
            for name in PATH_NAMES
        }
        measured: dict[str, dict[str, list[tuple[Any, ...]]]] = {
            name: {"naked": [], "observed": []} for name in PATH_NAMES
        }
        observed_logs: dict[str, list[list[dict[str, Any]]]] = {
            name: [] for name in PATH_NAMES
        }
        execution_order = []
        cycles = config.numerics.warmup_traces_per_shape + config.numerics.measured_traces_per_shape
        for cycle in range(cycles):
            path_rotation = cycle % len(PATH_NAMES)
            paths = PATH_NAMES[path_rotation:] + PATH_NAMES[:path_rotation]
            modes = ("naked", "observed") if cycle % 2 == 0 else ("observed", "naked")
            phase = "warm" if cycle < config.numerics.warmup_traces_per_shape else "measure"
            execution_order.append({"cycle": cycle, "phase": phase, "paths": list(paths), "modes": list(modes)})
            for path_name in paths:
                for mode in modes:
                    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
                    trace, observer = _run_trace(
                        trace_functions[path_name],
                        handle.model,
                        observed=mode == "observed",
                        capture_state=False,
                    )
                    if phase == "measure":
                        measured[path_name][mode].append(trace)
                        if observer is not None:
                            observed_logs[path_name].append(observer.calls)
                    print(
                        f"[observer-gate] {prompt['id']} {path_name} {mode} {phase} "
                        f"{cycle + 1}/{cycles}",
                        flush=True,
                    )
        prompt_result = {
            "prompt_length": int(input_ids.shape[-1]),
            "execution_order": execution_order,
            "paths": {},
        }
        prompt_tensors = {}
        for path_name in PATH_NAMES:
            naked = measured[path_name]["naked"]
            observed = measured[path_name]["observed"]
            naked_stability = _trace_metrics(naked[-2], naked[-1])
            observed_stability = _trace_metrics(observed[-2], observed[-1])
            inertness = _trace_metrics(observed[-1], naked[-1])
            call_stability = _call_log_diff(
                observed_logs[path_name][-2], observed_logs[path_name][-1]
            )
            passed = (
                naked_stability["exact_steps"] == len(FORCED_TOKEN_IDS)
                and observed_stability["exact_steps"] == len(FORCED_TOKEN_IDS)
                and inertness["exact_steps"] == len(FORCED_TOKEN_IDS)
                and call_stability["exact"]
            )
            if not passed:
                failures.append({"prompt": prompt["id"], "path": path_name})
            prompt_result["paths"][path_name] = {
                "passed": passed,
                "naked_stability": naked_stability,
                "observed_stability": observed_stability,
                "observed_vs_naked": inertness,
                "observed_call_stability": call_stability,
                "penultimate_observed_calls": observed_logs[path_name][-2],
                "final_observed_calls": observed_logs[path_name][-1],
            }
            prompt_tensors[path_name] = {
                "naked": torch.stack(list(naked[-1])),
                "observed": torch.stack(list(observed[-1])),
            }
        call_logs = {
            name: prompt_result["paths"][name]["final_observed_calls"]
            for name in PATH_NAMES
        }
        prompt_result["call_diffs"] = {
            "formic_runner_vs_hf_explicit": _call_log_diff(
                call_logs["formic_runner"], call_logs["hf_explicit"]
            ),
            "hf_generate_vs_hf_explicit": _call_log_diff(
                call_logs["hf_generate"], call_logs["hf_explicit"]
            ),
        }
        result["prompts"][prompt["id"]] = prompt_result
        tensor_payload[prompt["id"]] = prompt_tensors
    result["gate"] = {"passed": not failures, "failures": failures}
    _write_json("observer_gate.json", result)
    _write_tensors(
        "observer_gate.pt",
        {"config_hash": config.config_hash(), "tensors": tensor_payload},
    )
    if failures:
        raise RuntimeError(f"top-level observer is not bit-inert: {failures}")


def stage_state_gate(config_path: Path) -> None:
    """Capture state only after the independent observer gate passed."""
    from formic.science.determinism import configure_determinism

    observer_gate_path = ARTIFACT_DIR / "observer_gate.json"
    if not observer_gate_path.is_file():
        raise RuntimeError("observer gate artifact is missing; run --stage observer-gate first")
    observer_gate = json.loads(observer_gate_path.read_text(encoding="utf-8"))
    if not observer_gate.get("gate", {}).get("passed"):
        raise RuntimeError("observer gate did not pass; state capture is forbidden")

    torch = _torch()
    config, handle = _load_handle(config_path)
    _validate_protocol(config)
    state_metadata = _metadata(config, handle, "state_gate")
    if (
        observer_gate["metadata"]["protocol_identity_sha256"]
        != state_metadata["protocol_identity_sha256"]
    ):
        raise RuntimeError(
            "observer gate is stale or runtime-incompatible; rerun --stage observer-gate"
        )
    result = {
        "metadata": state_metadata,
        "protocol": {
            "paths": list(STATE_PATH_NAMES),
            "modes": ["naked", "state_captured"],
            "warmups_per_path_mode_prompt": config.numerics.warmup_traces_per_shape,
            "measured_per_path_mode_prompt": config.numerics.measured_traces_per_shape,
            "state_copy": "synchronous raw-byte CPU copy; no device clone retained",
            "path_order_alternates": True,
            "mode_order_alternates": True,
        },
        "prompts": {},
    }
    tensor_payload = {}
    failures = []
    for prompt in _render_prompts(handle.tokenizer, config):
        encoded = handle.tokenizer(prompt["text"], return_tensors="pt")
        device = _input_device(handle.model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        trace_functions = {
            name: _trace_function(name, handle, input_ids, attention_mask)
            for name in STATE_PATH_NAMES
        }
        measured: dict[str, dict[str, list[tuple[Any, ...]]]] = {
            name: {"naked": [], "state_captured": []} for name in STATE_PATH_NAMES
        }
        state_logs: dict[str, list[list[dict[str, Any]]]] = {
            name: [] for name in STATE_PATH_NAMES
        }
        execution_order = []
        cycles = config.numerics.warmup_traces_per_shape + config.numerics.measured_traces_per_shape
        for cycle in range(cycles):
            paths = STATE_PATH_NAMES if cycle % 2 == 0 else tuple(reversed(STATE_PATH_NAMES))
            modes = ("naked", "state_captured") if cycle % 2 == 0 else ("state_captured", "naked")
            phase = "warm" if cycle < config.numerics.warmup_traces_per_shape else "measure"
            execution_order.append({"cycle": cycle, "phase": phase, "paths": list(paths), "modes": list(modes)})
            for path_name in paths:
                for mode in modes:
                    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
                    trace, observer = _run_trace(
                        trace_functions[path_name],
                        handle.model,
                        observed=mode == "state_captured",
                        capture_state=mode == "state_captured",
                    )
                    if phase == "measure":
                        measured[path_name][mode].append(trace)
                        if observer is not None:
                            state_logs[path_name].append(observer.states)
                    print(
                        f"[state-gate] {prompt['id']} {path_name} {mode} {phase} "
                        f"{cycle + 1}/{cycles}",
                        flush=True,
                    )
        prompt_result = {
            "prompt_length": int(input_ids.shape[-1]),
            "execution_order": execution_order,
            "paths": {},
        }
        prompt_tensors = {}
        for path_name in STATE_PATH_NAMES:
            naked = measured[path_name]["naked"]
            captured = measured[path_name]["state_captured"]
            naked_stability = _trace_metrics(naked[-2], naked[-1])
            captured_stability = _trace_metrics(captured[-2], captured[-1])
            inertness = _trace_metrics(captured[-1], naked[-1])
            state_stability = _state_component_diff(
                state_logs[path_name][-2], state_logs[path_name][-1]
            )
            penultimate_completeness = _validate_state_trace(
                state_logs[path_name][-2], int(input_ids.shape[-1])
            )
            final_completeness = _validate_state_trace(
                state_logs[path_name][-1], int(input_ids.shape[-1])
            )
            passed = (
                naked_stability["exact_steps"] == len(FORCED_TOKEN_IDS)
                and captured_stability["exact_steps"] == len(FORCED_TOKEN_IDS)
                and inertness["exact_steps"] == len(FORCED_TOKEN_IDS)
                and state_stability["exact"]
            )
            if not passed:
                failures.append({"prompt": prompt["id"], "path": path_name})
            prompt_result["paths"][path_name] = {
                "passed": passed,
                "naked_stability": naked_stability,
                "captured_stability": captured_stability,
                "captured_vs_naked": inertness,
                "captured_state_stability": state_stability,
                "penultimate_state_completeness": penultimate_completeness,
                "final_state_completeness": final_completeness,
                "final_state": state_logs[path_name][-1],
            }
            prompt_tensors[path_name] = {
                "naked": torch.stack(list(naked[-1])),
                "state_captured": torch.stack(list(captured[-1])),
            }
        runner_trace = measured["formic_runner"]["state_captured"][-1]
        explicit_trace = measured["hf_explicit"]["state_captured"][-1]
        logits_diff = _trace_metrics(runner_trace, explicit_trace)
        state_diff = _state_component_diff(
            state_logs["formic_runner"][-1], state_logs["hf_explicit"][-1]
        )
        first_state_index = (
            state_diff["first_difference"]["boundary_index"]
            if state_diff["first_difference"] is not None
            else None
        )
        first_logit_index = logits_diff["first_divergence"]
        prompt_result["formic_runner_vs_hf_explicit"] = {
            "logits": logits_diff,
            "state": state_diff,
            "first_state_divergence_precedes_first_logit_divergence": (
                first_state_index is not None
                and first_logit_index is not None
                and first_state_index < first_logit_index
            ),
        }
        result["prompts"][prompt["id"]] = prompt_result
        tensor_payload[prompt["id"]] = prompt_tensors
    result["gate"] = {"passed": not failures, "failures": failures}
    _write_json("state_gate.json", result)
    _write_tensors(
        "state_gate.pt",
        {"config_hash": config.config_hash(), "tensors": tensor_payload},
    )
    if failures:
        raise RuntimeError(f"state observer is not bit-inert or stable: {failures}")


def _summary_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['exact_steps']}/{metric['steps']} exact, "
        f"top-1 {metric['top1_matches']}/{metric['steps']}, "
        f"first={metric['first_divergence_boundary']}, "
        f"max delta={metric['max_abs_logit_delta']:.6e}"
    )


def stage_report() -> None:
    observer_path = ARTIFACT_DIR / "observer_gate.json"
    state_path = ARTIFACT_DIR / "state_gate.json"
    if not observer_path.is_file() or not state_path.is_file():
        raise RuntimeError("observer_gate.json and state_gate.json are required")
    observer = json.loads(observer_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not observer["gate"]["passed"] or not state["gate"]["passed"]:
        raise RuntimeError("cannot report a failed observer/state gate")
    if (
        observer["metadata"]["protocol_identity_sha256"]
        != state["metadata"]["protocol_identity_sha256"]
    ):
        raise RuntimeError("diagnostic artifact protocol identities differ")

    lines = [
        "# SPEC-01 runner call/state diagnostics",
        "",
        "Diagnostic only. No runner correction or causal attribution is made here. ",
        "SPEC-01 remains 8/9 and ADR-0004 remains PROPOSED.",
        "",
        "## Protocol",
        "",
        f"- Config hash: `{observer['metadata']['config_hash']}`",
        f"- Prompt-set hash: `{observer['metadata']['prompt_set_sha256']}`",
        "- Text only, batch 1, BF16, six warmups and two measured traces.",
        "- A naked/observed bit-inertness gate ran first on all three paths and all six prompts.",
        "- State hashing then ran as a separate naked/captured gate on runner and explicit HF.",
        "- Boundaries: `prefill` produces legacy logit step 0; `after_forced_N` produces legacy step N+1.",
        "",
        "## Observer gate",
        "",
        f"Result: **{'PASS' if observer['gate']['passed'] else 'FAIL'}**.",
        "",
        "| Prompt | Path | Observed vs naked |",
        "|---|---|---|",
    ]
    for prompt_id, prompt in observer["prompts"].items():
        for path_name in PATH_NAMES:
            metric = prompt["paths"][path_name]["observed_vs_naked"]
            lines.append(f"| `{prompt_id}` | `{path_name}` | {_summary_metric(metric)} |")

    lines += [
        "",
        "## Call arguments",
        "",
        "| Prompt | Runner vs explicit | Generate vs explicit |",
        "|---|---|---|",
    ]
    for prompt_id, prompt in observer["prompts"].items():
        runner = prompt["call_diffs"]["formic_runner_vs_hf_explicit"]
        generate = prompt["call_diffs"]["hf_generate_vs_hf_explicit"]
        runner_first = runner["first_difference"]["field"] if runner["first_difference"] else "none"
        generate_first = generate["first_difference"]["field"] if generate["first_difference"] else "none"
        lines.append(
            f"| `{prompt_id}` | {runner['difference_count']} differences; first `{runner_first}` | "
            f"{generate['difference_count']} differences; first `{generate_first}` |"
        )

    lines += [
        "",
        "## State gate",
        "",
        f"Result: **{'PASS' if state['gate']['passed'] else 'FAIL'}**.",
        "",
        "| Prompt | State-captured vs naked runner | State-captured vs naked explicit |",
        "|---|---|---|",
    ]
    for prompt_id, prompt in state["prompts"].items():
        runner = prompt["paths"]["formic_runner"]["captured_vs_naked"]
        explicit = prompt["paths"]["hf_explicit"]["captured_vs_naked"]
        lines.append(
            f"| `{prompt_id}` | {_summary_metric(runner)} | {_summary_metric(explicit)} |"
        )

    lines += [
        "",
        "## First divergences",
        "",
        "| Prompt | First logit | First state | Component | State precedes logit |",
        "|---|---|---|---|---|",
    ]
    for prompt_id, prompt in state["prompts"].items():
        comparison = prompt["formic_runner_vs_hf_explicit"]
        first_logit = comparison["logits"]["first_divergence_boundary"]
        first_state = comparison["state"]["first_difference"]
        if first_state is None:
            state_boundary = component = "none"
        else:
            state_boundary = first_state["boundary"]
            component = f"layer {first_state['layer']} / {first_state['component']}"
        lines.append(
            f"| `{prompt_id}` | `{first_logit}` | `{state_boundary}` | `{component}` | "
            f"{comparison['first_state_divergence_precedes_first_logit_divergence']} |"
        )

    lines += [
        "",
        "## Generate convention",
        "",
        "The exact runtime argument records, including absent/default distinctions, tensor hashes, "
        "attention masks, position IDs, cache metadata and `logits_to_keep`, are in "
        "`artifacts/step1/runner_state_diagnostics/observer_gate.json`. This report intentionally "
        "does not infer causality from those differences.",
        "",
        "## Environment",
        "",
        f"- Torch: `{observer['metadata']['environment'].get('torch')}`",
        f"- CUDA: `{observer['metadata']['environment'].get('cuda_version')}`",
        f"- GPUs: `{json.dumps(observer['metadata']['environment'].get('gpus', []))}`",
        f"- `pip freeze` SHA-256: `{observer['metadata']['pip_freeze']['sha256']}`",
        "",
    ]
    path = REPO_ROOT / "reports" / "step1_runner_state_diagnostics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path.relative_to(REPO_ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 runner call/state diagnostics")
    parser.add_argument(
        "--stage",
        choices=("observer-gate", "state-gate", "report"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    prepare_backend_environment(load_config(args.config).numerics)
    if args.stage == "observer-gate":
        stage_observer_gate(args.config)
    elif args.stage == "state-gate":
        stage_state_gate(args.config)
    else:
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
