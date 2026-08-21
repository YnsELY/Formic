"""Turn aligned forward traces into per-point, per-step measurements."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from formic.science.identity.metrics import compare_logits, compare_tensors
from formic.science.identity.trace import ForwardTrace
from formic.science.identity.types import (
    ComparisonLocation,
    ComparisonPoint,
    ExecutionMode,
    InputShape,
)
from formic.science.identity.verdict import MeasuredComparison
from formic.state.snapshot import (
    AttentionLayerSnapshot,
    ExecutionSnapshot,
    GdnLayerSnapshot,
    ModelStateSlot,
)


class TraceStructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceComparison:
    measurements: tuple[MeasuredComparison, ...]
    not_applicable: tuple[ComparisonLocation, ...]


def _scalar(value) -> torch.Tensor:
    if isinstance(value, bool):
        return torch.tensor([value], dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor([value], dtype=torch.int64)
    if isinstance(value, float):
        return torch.tensor([value], dtype=torch.float64)
    raise TraceStructureError(f"unsupported scalar state {type(value)!r}")


def compare_forward_traces(
    reference: ForwardTrace,
    candidate: ForwardTrace,
    *,
    step: int,
    mode: ExecutionMode,
    length_class: str,
    input_shape: InputShape,
) -> TraceComparison:
    measurements: list[MeasuredComparison] = []
    not_applicable: list[ComparisonLocation] = []

    def add(location: ComparisonLocation, metric) -> None:
        measurements.append(
            MeasuredComparison(step, mode, length_class, input_shape, location, metric)
        )

    add(ComparisonLocation(ComparisonPoint.LOGITS), compare_logits(reference.logits, candidate.logits))
    if len(reference.boundaries) != len(candidate.boundaries):
        raise TraceStructureError("reference/candidate boundary counts differ")
    for ref_boundary, cand_boundary in zip(reference.boundaries, candidate.boundaries):
        if ref_boundary.name != cand_boundary.name:
            raise TraceStructureError(
                f"boundary order differs: {ref_boundary.name} != {cand_boundary.name}"
            )
        add(
            ComparisonLocation(ComparisonPoint.HIDDEN_STATE, ref_boundary.name),
            compare_tensors(ref_boundary.hidden_states, cand_boundary.hidden_states),
        )
        if ref_boundary.cache_applicability != cand_boundary.cache_applicability:
            raise TraceStructureError(f"cache applicability differs at {ref_boundary.name}")
        if ref_boundary.cache_applicability == "not_applicable":
            if ref_boundary.completed_group is not None:
                not_applicable.extend(
                    (
                        ComparisonLocation(ComparisonPoint.GDN_STATE, ref_boundary.name),
                        ComparisonLocation(ComparisonPoint.ATTENTION_KV, ref_boundary.name),
                    )
                )
            continue
        _compare_layers(ref_boundary.cache_layers, cand_boundary.cache_layers, ref_boundary.name, add)

    _compare_model_state(reference.model_state, candidate.model_state, add)
    if (reference.final_state is None) != (candidate.final_state is None):
        raise TraceStructureError("final-state applicability differs")
    if reference.final_state is not None and candidate.final_state is not None:
        _compare_final_state(reference.final_state, candidate.final_state, add)
    return TraceComparison(tuple(measurements), tuple(not_applicable))


def _compare_layers(reference, candidate, boundary, add) -> None:
    if len(reference) != len(candidate):
        raise TraceStructureError(f"cache layer count differs at {boundary}")
    for ref_layer, cand_layer in zip(reference, candidate):
        if ref_layer.layer_index != cand_layer.layer_index or type(ref_layer) is not type(cand_layer):
            raise TraceStructureError(f"cache layer structure differs at {boundary}")
        layer = ref_layer.layer_index
        if isinstance(ref_layer, GdnLayerSnapshot):
            assert isinstance(cand_layer, GdnLayerSnapshot)
            point = ComparisonPoint.GDN_STATE
            _compare_optional_tensor(ref_layer.conv_states, cand_layer.conv_states, point, boundary, layer, "conv", add)
            _compare_optional_tensor(
                ref_layer.recurrent_states, cand_layer.recurrent_states,
                point, boundary, layer, "recurrent", add,
            )
            for component in (
                "is_conv_states_initialized",
                "is_recurrent_states_initialized",
                "has_previous_state",
            ):
                add(
                    ComparisonLocation(point, boundary, layer, component),
                    compare_tensors(_scalar(getattr(ref_layer, component)), _scalar(getattr(cand_layer, component))),
                )
        elif isinstance(ref_layer, AttentionLayerSnapshot):
            assert isinstance(cand_layer, AttentionLayerSnapshot)
            point = ComparisonPoint.ATTENTION_KV
            _compare_optional_tensor(ref_layer.keys, cand_layer.keys, point, boundary, layer, "keys", add)
            _compare_optional_tensor(ref_layer.values, cand_layer.values, point, boundary, layer, "values", add)
            for component in ("is_initialized", "sequence_length"):
                add(
                    ComparisonLocation(point, boundary, layer, component),
                    compare_tensors(_scalar(getattr(ref_layer, component)), _scalar(getattr(cand_layer, component))),
                )


def _compare_optional_tensor(reference, candidate, point, boundary, layer, component, add):
    if (reference is None) != (candidate is None):
        add(
            ComparisonLocation(point, boundary, layer, f"{component}_present"),
            compare_tensors(_scalar(reference is not None), _scalar(candidate is not None)),
        )
    elif reference is not None and candidate is not None:
        add(
            ComparisonLocation(point, boundary, layer, component),
            compare_tensors(reference, candidate),
        )


def _compare_model_state(reference: tuple[ModelStateSlot, ...], candidate: tuple[ModelStateSlot, ...], add) -> None:
    ref_keys = tuple((item.module_path, item.attribute) for item in reference)
    cand_keys = tuple((item.module_path, item.attribute) for item in candidate)
    if ref_keys != cand_keys:
        raise TraceStructureError("model-attached state registry differs")
    for ref_slot, cand_slot in zip(reference, candidate):
        component = f"{ref_slot.module_path or 'absent'}.{ref_slot.attribute}"
        statuses = {"absent": 0, "none": 1, "tensor": 2, "scalar": 3}
        add(
            ComparisonLocation(ComparisonPoint.MODEL_STATE, component=f"{component}.status"),
            compare_tensors(_scalar(statuses[ref_slot.status]), _scalar(statuses[cand_slot.status])),
        )
        if ref_slot.status != cand_slot.status:
            continue
        elif isinstance(ref_slot.value, torch.Tensor) and isinstance(cand_slot.value, torch.Tensor):
            add(
                ComparisonLocation(ComparisonPoint.MODEL_STATE, component=component),
                compare_tensors(ref_slot.value, cand_slot.value),
            )
        elif ref_slot.value is not None and cand_slot.value is not None:
            add(
                ComparisonLocation(ComparisonPoint.MODEL_STATE, component=component),
                compare_tensors(_scalar(ref_slot.value), _scalar(cand_slot.value)),
            )


def _compare_final_state(reference: ExecutionSnapshot, candidate: ExecutionSnapshot, add) -> None:
    _compare_layers(reference.layers, candidate.layers, "FINAL", add)
    for component in ("sequence_length",):
        add(
            ComparisonLocation(ComparisonPoint.MODEL_STATE, component=f"position.{component}"),
            compare_tensors(_scalar(getattr(reference.position, component)), _scalar(getattr(candidate.position, component))),
        )
    for component in ("cache_position", "position_ids", "attention_mask"):
        _compare_optional_tensor(
            getattr(reference.position, component), getattr(candidate.position, component),
            ComparisonPoint.MODEL_STATE, None, None, f"position.{component}", add,
        )
