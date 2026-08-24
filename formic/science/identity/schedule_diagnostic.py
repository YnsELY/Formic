"""Memory-bounded execution calendars for isolated SPEC-02 diagnostics.

This module is deliberately isolated from the production identity executor. It
uses stock model forwards and fresh configured caches (A1/A2), never restores or
shares cache state (A3/A4), and serialises only CPU scalars and content hashes.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, Callable, Literal

import torch

from formic.backbone.runner import identity_forward
from formic.science.identity.executor import Endpoint
from formic.science.identity.metrics import compare_logits


Calendar = Literal["sequential", "alternating", "abba", "baab"]
ScalarObserver = Callable[[dict[str, Any]], None]
MemoryObserver = Callable[[str], None]
CPULogitsObserver = Callable[[dict[str, Any], torch.Tensor], None]


@torch.no_grad()
def run_schedule_pair(
    calendar: Calendar,
    left: Endpoint,
    right: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    forced_token_ids: tuple[int, ...],
    capture: bool,
    event_observer: ScalarObserver | None = None,
    memory_observer: MemoryObserver | None = None,
    cpu_logits_observer: CPULogitsObserver | None = None,
) -> dict[str, Any] | None:
    """Run one fresh-cache pair without retaining CUDA outputs or traces."""
    if calendar not in ("sequential", "alternating", "abba", "baab"):
        raise ValueError(f"unknown calendar: {calendar}")
    if not forced_token_ids:
        raise ValueError("cached decode requires a forced continuation")
    if torch.is_grad_enabled():  # pragma: no cover - enforced by decorator
        raise RuntimeError("schedule diagnostic requires autograd disabled")

    from transformers.cache_utils import DynamicCache

    if memory_observer is not None:
        memory_observer("before_cache_creation")
    left_cache = DynamicCache(config=left.model.config)
    right_cache = DynamicCache(config=right.model.config)
    cache_object_ids = {"left": id(left_cache), "right": id(right_cache)}
    if left_cache is right_cache:
        raise RuntimeError("diagnostic endpoints share a cache object")

    devices = {
        "left": next(left.model.parameters()).device,
        "right": next(right.model.parameters()).device,
    }
    inputs = {
        "left": torch.tensor([prompt_token_ids], dtype=torch.long, device=devices["left"]),
        "right": torch.tensor([prompt_token_ids], dtype=torch.long, device=devices["right"]),
    }
    endpoints = {"left": left, "right": right}
    caches = {"left": left_cache, "right": right_cache}
    cpu_logits: dict[str, list[torch.Tensor]] = {"left": [], "right": []}
    records: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    forward_order: list[dict[str, Any]] = []
    cache_storage_disjoint = True
    autograd_disabled = True

    order = _forward_order(calendar, len(forced_token_ids))
    try:
        for pair_local_ordinal, (side, step, within_step_ordinal) in enumerate(order):
            current_grad = torch.is_grad_enabled()
            autograd_disabled = autograd_disabled and not current_grad
            if current_grad:
                raise RuntimeError(f"autograd enabled for {side} step {step}")
            output = _call_endpoint(endpoints[side], inputs[side], caches[side])
            metadata = _forward_metadata(
                calendar=calendar,
                endpoint=endpoints[side].name,
                side=side,
                step=step,
                pair_local_ordinal=pair_local_ordinal,
                within_step_ordinal=within_step_ordinal,
            )
            forward_order.append(metadata)
            _observe(
                event_observer,
                event="after_endpoint",
                **metadata,
                grad_enabled=current_grad,
                cache_object_id=cache_object_ids[side],
            )
            if memory_observer is not None:
                memory_observer(f"after_{side}_step_{step}")

            if capture or cpu_logits_observer is not None:
                logits_cpu = output.logits[0, -1].detach().to("cpu").contiguous()
                if cpu_logits_observer is not None:
                    cpu_logits_observer(dict(metadata), logits_cpu)
                if capture:
                    cpu_logits[side].append(logits_cpu)
                    records[side].append(_logit_record(logits_cpu, metadata))
            del output
            _observe(
                event_observer,
                event="after_output_deleted",
                **metadata,
                grad_enabled=torch.is_grad_enabled(),
                cache_object_id=cache_object_ids[side],
            )
            if memory_observer is not None:
                memory_observer(f"after_{side}_step_{step}_output_deleted")

            if calendar == "sequential" or within_step_ordinal == 1:
                left_storage = _cache_storage_pointers(left_cache)
                right_storage = _cache_storage_pointers(right_cache)
                cache_storage_disjoint = cache_storage_disjoint and not bool(
                    left_storage & right_storage
                )
                if not cache_storage_disjoint:
                    raise RuntimeError("diagnostic endpoint cache storages overlap")

            if step < len(forced_token_ids) - 1:
                del inputs[side]
                inputs[side] = torch.tensor(
                    [[forced_token_ids[step]]], dtype=torch.long, device=devices[side]
                )

        if not capture:
            return None

        steps: list[dict[str, Any]] = []
        for step, (left_logits, right_logits) in enumerate(
            zip(cpu_logits["left"], cpu_logits["right"])
        ):
            metric = compare_logits(left_logits, right_logits).to_dict()
            steps.append(
                {
                    "step": step,
                    "point": "logits",
                    "left": records["left"][step],
                    "right": records["right"][step],
                    "comparison": {
                        "max_abs_delta": metric["tensor"]["max_abs_delta"],
                        "kl_next_token": metric["kl_next_token"],
                        "left_top1": metric["reference_top1"],
                        "right_top1": metric["candidate_top1"],
                        "top1_agreement": metric["top1_agreement"],
                        "exact": metric["tensor"]["exact"],
                        "first_coordinate": metric["tensor"]["first_coordinate"],
                        "left_value": metric["tensor"]["reference_value"],
                        "right_value": metric["tensor"]["candidate_value"],
                    },
                }
            )
        result = {
            "left_endpoint": left.name,
            "right_endpoint": right.name,
            "left_path_fingerprint": _path_fingerprint(left.name, records["left"]),
            "right_path_fingerprint": _path_fingerprint(right.name, records["right"]),
            "forward_order": forward_order,
            "autograd_disabled_all_forwards": autograd_disabled,
            "cache_independence": {
                "cache_objects_distinct": cache_object_ids["left"] != cache_object_ids["right"],
                "cache_storage_disjoint": cache_storage_disjoint,
                "fresh_cache_pair_constructed_for_call": True,
            },
            "steps": steps,
        }
        if _contains_tensor(result):  # pragma: no cover - defensive serialization gate
            raise RuntimeError("diagnostic result retained a tensor")
        return result
    finally:
        cpu_logits["left"].clear()
        cpu_logits["right"].clear()
        records["left"].clear()
        records["right"].clear()
        inputs.clear()
        caches.clear()
        del left_cache
        del right_cache
        if memory_observer is not None:
            memory_observer("after_warmup" if not capture else "after_measurement")


def _call_endpoint(endpoint: Endpoint, input_ids: torch.Tensor, cache: Any) -> Any:
    if torch.is_grad_enabled():
        raise RuntimeError("autograd enabled inside diagnostic forward")
    kwargs = {"input_ids": input_ids, "past_key_values": cache, "use_cache": True}
    if endpoint.through_formic_runner:
        return identity_forward(SimpleNamespace(model=endpoint.model), **kwargs)
    return endpoint.model(**kwargs)


def _forward_order(calendar: Calendar, steps: int) -> list[tuple[str, int, int]]:
    if calendar == "sequential":
        return [
            (side, step, 0 if side == "left" else 1)
            for side in ("left", "right")
            for step in range(steps)
        ]
    if calendar == "alternating":
        first_sides = ("left",) * steps
    else:
        pattern = (
            ("left", "right", "right", "left")
            if calendar == "abba"
            else ("right", "left", "left", "right")
        )
        first_sides = tuple(pattern[step % len(pattern)] for step in range(steps))
    return [
        (side, step, within)
        for step, first in enumerate(first_sides)
        for within, side in enumerate((first, "right" if first == "left" else "left"))
    ]


def _forward_metadata(
    *,
    calendar: Calendar,
    endpoint: str,
    side: str,
    step: int,
    pair_local_ordinal: int,
    within_step_ordinal: int,
) -> dict[str, Any]:
    return {
        "calendar": calendar,
        "endpoint": endpoint,
        "side": side,
        "decode_step": step,
        "step": step,
        "pair_local_forward_ordinal": pair_local_ordinal,
        "within_step_ordinal": within_step_ordinal,
    }


def _logit_record(logits_cpu: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
    if logits_cpu.device.type != "cpu" or logits_cpu.requires_grad:
        raise RuntimeError("diagnostic logits must be detached on CPU")
    raw = logits_cpu.reshape(-1).view(torch.uint8)
    return {
        **metadata,
        "sha256": hashlib.sha256(memoryview(raw.numpy())).hexdigest(),
        "top1": int(torch.argmax(logits_cpu).item()),
        "shape": list(logits_cpu.shape),
        "dtype": str(logits_cpu.dtype),
        "device": "cpu",
    }


def _path_fingerprint(endpoint: str, records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"endpoint": endpoint, "steps": records}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_storage_pointers(cache: Any) -> set[int]:
    pointers: set[int] = set()
    for layer in getattr(cache, "layers", ()):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                pointers.add(int(value.untyped_storage().data_ptr()))
    return pointers


def _observe(observer: ScalarObserver | None, **event: Any) -> None:
    if observer is not None:
        observer(event)


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False
