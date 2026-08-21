"""In-memory snapshot/restore for the audited hybrid Qwen cache.

Audit constraints engaged: A1 (``use_cache=False`` is not write protection),
A2 (restore always builds ``DynamicCache(config=model.config)``), A3 (no
``crop``), A4 (deep clone per consumer), A6 (model state is separate), A8
(batch 1), and A9 (K/V are copied exactly as stored).

Snapshots are memory-only in SPEC-02 but contain only explicit dataclasses,
primitive metadata and tensors: no model reference, cache-layer object or
closure. They are therefore serialisable in principle without implementing
disk persistence in this step.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterator, Literal

import torch

from formic.backbone import constants as C

__all__ = [
    "AttentionLayerSnapshot",
    "BranchActivationError",
    "ExecutionSnapshot",
    "ExecutionStateController",
    "GdnLayerSnapshot",
    "ModelStateSlot",
    "PositionState",
    "RestoredExecutionState",
    "SnapshotError",
    "capture_cache_layers",
    "capture_model_state",
    "iter_snapshot_tensors",
    "restore",
    "snapshot",
    "snapshot_fingerprint",
    "tensor_storage_identity",
]

SNAPSHOT_SCHEMA_VERSION = 1
MODEL_STATE_ATTRIBUTES = ("rope_deltas",)


class SnapshotError(RuntimeError):
    """The supplied state is incomplete or violates the audited cache layout."""


class BranchActivationError(RuntimeError):
    """A restored branch was executed without being the active branch."""


@dataclass(frozen=True)
class PositionState:
    sequence_length: int
    cache_position: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GdnLayerSnapshot:
    layer_index: int
    layer_type: str
    conv_states: torch.Tensor | None
    recurrent_states: torch.Tensor | None
    is_conv_states_initialized: bool
    is_recurrent_states_initialized: bool
    has_previous_state: bool
    dtype: str | None
    device: str | None


@dataclass(frozen=True)
class AttentionLayerSnapshot:
    layer_index: int
    layer_type: str
    keys: torch.Tensor | None
    values: torch.Tensor | None
    is_initialized: bool
    sequence_length: int
    dtype: str | None
    device: str | None


LayerSnapshot = GdnLayerSnapshot | AttentionLayerSnapshot
ModelSlotStatus = Literal["absent", "none", "tensor", "scalar"]


@dataclass(frozen=True)
class ModelStateSlot:
    module_path: str | None
    attribute: str
    status: ModelSlotStatus
    value: torch.Tensor | bool | int | float | str | None


@dataclass(frozen=True)
class ExecutionSnapshot:
    schema_version: int
    cache_type: str
    cache_offloading: bool
    cache_offload_only_non_sliding: bool
    layers: tuple[LayerSnapshot, ...]
    position: PositionState
    model_state: tuple[ModelStateSlot, ...]


@dataclass
class RestoredExecutionState:
    """One live consumer restored from an immutable snapshot."""

    cache: Any
    position: PositionState
    model_state: tuple[ModelStateSlot, ...]
    branch_id: str
    _controller_id: str
    _activation_epoch: int = 0


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _clone_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().clone(memory_format=torch.preserve_format)


def _clone_position(position: PositionState) -> PositionState:
    if position.sequence_length < 0:
        raise SnapshotError("position.sequence_length must be non-negative")
    return PositionState(
        sequence_length=position.sequence_length,
        cache_position=_clone_tensor(position.cache_position),
        position_ids=_clone_tensor(position.position_ids),
        attention_mask=_clone_tensor(position.attention_mask),
    )


def capture_model_state(model: Any) -> tuple[ModelStateSlot, ...]:
    """Deep-clone the explicit registry of model-attached execution state (A6)."""
    slots: list[ModelStateSlot] = []
    modules = list(model.named_modules())
    for attribute in MODEL_STATE_ATTRIBUTES:
        found = False
        for module_path, module in modules:
            if attribute not in vars(module):
                continue
            found = True
            value = vars(module)[attribute]
            path = module_path or "<root>"
            if value is None:
                slots.append(ModelStateSlot(path, attribute, "none", None))
            elif isinstance(value, torch.Tensor):
                slots.append(ModelStateSlot(path, attribute, "tensor", _clone_tensor(value)))
            elif isinstance(value, (bool, int, float, str)):
                slots.append(ModelStateSlot(path, attribute, "scalar", value))
            else:
                raise SnapshotError(
                    f"unsupported model state {path}.{attribute}: {type(value)!r}"
                )
        if not found:
            slots.append(ModelStateSlot(None, attribute, "absent", None))
    return tuple(slots)


def _clone_model_state(state: tuple[ModelStateSlot, ...]) -> tuple[ModelStateSlot, ...]:
    return tuple(
        ModelStateSlot(
            slot.module_path,
            slot.attribute,
            slot.status,
            _clone_tensor(slot.value) if isinstance(slot.value, torch.Tensor) else slot.value,
        )
        for slot in state
    )


def _capture_layer(index: int, layer: Any) -> LayerSnapshot:
    layer_type = _qualified_type(layer)
    if index in C.ATTENTION_LAYER_INDICES:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        initialized = bool(getattr(layer, "is_initialized", False))
        if (keys is None) != (values is None):
            raise SnapshotError(f"attention layer {index} has incomplete K/V")
        if initialized != (keys is not None and values is not None):
            raise SnapshotError(f"attention layer {index} flag/tensors disagree")
        try:
            length = int(layer.get_seq_length())
        except (AttributeError, TypeError, ValueError) as exc:
            raise SnapshotError(f"invalid attention length at layer {index}") from exc
        return AttentionLayerSnapshot(
            index,
            layer_type,
            _clone_tensor(keys),
            _clone_tensor(values),
            initialized,
            length,
            str(keys.dtype) if keys is not None else None,
            str(keys.device) if keys is not None else None,
        )

    conv = getattr(layer, "conv_states", None)
    recurrent = getattr(layer, "recurrent_states", None)
    conv_initialized = bool(getattr(layer, "is_conv_states_initialized", False))
    recurrent_initialized = bool(getattr(layer, "is_recurrent_states_initialized", False))
    if conv_initialized != (conv is not None):
        raise SnapshotError(f"GDN layer {index} conv flag disagrees")
    if recurrent_initialized != (recurrent is not None):
        raise SnapshotError(f"GDN layer {index} recurrent flag disagrees")
    exemplar = conv if conv is not None else recurrent
    return GdnLayerSnapshot(
        index,
        layer_type,
        _clone_tensor(conv),
        _clone_tensor(recurrent),
        conv_initialized,
        recurrent_initialized,
        bool(getattr(layer, "has_previous_state", False)),
        str(exemplar.dtype) if exemplar is not None else None,
        str(exemplar.device) if exemplar is not None else None,
    )


def capture_cache_layers(cache: Any, layer_indices: Iterator[int] | tuple[int, ...]) -> tuple[LayerSnapshot, ...]:
    """Deep-clone selected cache layers without duplicating the cumulative cache."""
    layers = list(getattr(cache, "layers", ()))
    if len(layers) != C.NUM_LAYERS:
        raise SnapshotError(f"expected {C.NUM_LAYERS} cache layers, got {len(layers)}")
    indices = tuple(layer_indices)
    if len(indices) != len(set(indices)):
        raise SnapshotError("cache layer indices must be unique")
    if any(index < 0 or index >= C.NUM_LAYERS for index in indices):
        raise SnapshotError("cache layer index out of range")
    return tuple(_capture_layer(index, layers[index]) for index in indices)


def _capture_layers(cache: Any) -> tuple[LayerSnapshot, ...]:
    return capture_cache_layers(cache, tuple(range(C.NUM_LAYERS)))


def snapshot(*, model: Any, cache: Any, position: PositionState) -> ExecutionSnapshot:
    """Capture cache, positions and model-attached state by deep clone."""
    if model is None or not hasattr(model, "config"):
        raise SnapshotError("snapshot requires a live model exposing config")
    try:
        cache_length = int(cache.get_seq_length())
    except (AttributeError, TypeError, ValueError) as exc:
        raise SnapshotError("snapshot requires a valid hybrid cache") from exc
    if cache_length != position.sequence_length:
        raise SnapshotError(
            f"position length {position.sequence_length} != cache length {cache_length}"
        )
    offloading = bool(getattr(cache, "offloading", False))
    return ExecutionSnapshot(
        SNAPSHOT_SCHEMA_VERSION,
        _qualified_type(cache),
        offloading,
        bool(getattr(cache, "only_non_sliding", False)) if offloading else False,
        _capture_layers(cache),
        _clone_position(position),
        capture_model_state(model),
    )


def _validate_layout(value: ExecutionSnapshot) -> None:
    if value.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(
            f"snapshot schema {value.schema_version} != {SNAPSHOT_SCHEMA_VERSION}"
        )
    if len(value.layers) != C.NUM_LAYERS:
        raise SnapshotError(f"snapshot has {len(value.layers)} layers")
    for index, layer in enumerate(value.layers):
        if layer.layer_index != index:
            raise SnapshotError(f"snapshot layer ordering changed at {index}")
        if index in C.ATTENTION_LAYER_INDICES:
            if not isinstance(layer, AttentionLayerSnapshot):
                raise SnapshotError(f"layer {index} must be full attention")
        elif not isinstance(layer, GdnLayerSnapshot):
            raise SnapshotError(f"layer {index} must be GDN")


def _restore_attention(target: Any, source: AttentionLayerSnapshot) -> None:
    if not source.is_initialized:
        if source.keys is not None or source.values is not None or source.sequence_length != 0:
            raise SnapshotError(f"uninitialized attention {source.layer_index} has state")
        return
    if source.keys is None or source.values is None:
        raise SnapshotError(f"attention {source.layer_index} has incomplete K/V")
    target.update(_clone_tensor(source.keys), _clone_tensor(source.values))
    if int(target.get_seq_length()) != source.sequence_length:
        raise SnapshotError(f"attention {source.layer_index} length mismatch")


def _restore_gdn(target: Any, source: GdnLayerSnapshot) -> None:
    if source.is_conv_states_initialized:
        if source.conv_states is None:
            raise SnapshotError(f"GDN {source.layer_index} has no conv state")
        conv = _clone_tensor(source.conv_states)
        target.lazy_initialization(conv_states=conv)
        target.conv_states.copy_(conv)
    elif source.conv_states is not None:
        raise SnapshotError(f"GDN {source.layer_index} has stray conv state")

    if source.is_recurrent_states_initialized:
        if source.recurrent_states is None:
            raise SnapshotError(f"GDN {source.layer_index} has no recurrent state")
        recurrent = _clone_tensor(source.recurrent_states)
        if not hasattr(target, "dtype"):
            target.dtype, target.device = recurrent.dtype, recurrent.device
        target.lazy_initialization(recurrent_states=recurrent)
        target.recurrent_states.copy_(recurrent)
    elif source.recurrent_states is not None:
        raise SnapshotError(f"GDN {source.layer_index} has stray recurrent state")
    target.has_previous_state = source.has_previous_state


def restore(
    *, model: Any, snapshot: ExecutionSnapshot
) -> tuple[Any, PositionState, tuple[ModelStateSlot, ...]]:
    """Materialise one independent cache consumer from ``snapshot``."""
    if model is None or not hasattr(model, "config"):
        raise SnapshotError("restore requires a live model exposing config")
    _validate_layout(snapshot)
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(
        config=model.config,
        offloading=snapshot.cache_offloading,
        offload_only_non_sliding=snapshot.cache_offload_only_non_sliding,
    )
    if len(cache.layers) != C.NUM_LAYERS:
        raise SnapshotError(f"model config created {len(cache.layers)} cache layers")
    for index, source in enumerate(snapshot.layers):
        target = cache.layers[index]
        if _qualified_type(target) != source.layer_type:
            raise SnapshotError(f"cache layer type changed at {index}")
        if isinstance(source, AttentionLayerSnapshot):
            _restore_attention(target, source)
        else:
            _restore_gdn(target, source)
    position = _clone_position(snapshot.position)
    if int(cache.get_seq_length()) != position.sequence_length:
        raise SnapshotError("restored cache length differs from position state")
    return cache, position, _clone_model_state(snapshot.model_state)


def _apply_model_state(model: Any, state: tuple[ModelStateSlot, ...]) -> None:
    modules = {(path or "<root>"): module for path, module in model.named_modules()}
    for slot in state:
        if slot.status == "absent":
            if slot.module_path is not None:
                raise SnapshotError("absent model-state slot has a module path")
            continue
        if slot.module_path not in modules:
            raise SnapshotError(f"model-state module missing: {slot.module_path!r}")
        module = modules[slot.module_path]
        if slot.attribute not in vars(module):
            raise SnapshotError(f"model-state attribute missing: {slot.attribute}")
        if slot.status == "none":
            setattr(module, slot.attribute, None)
        elif slot.status == "tensor":
            if not isinstance(slot.value, torch.Tensor):
                raise SnapshotError("tensor model-state slot contains non-tensor")
            setattr(module, slot.attribute, _clone_tensor(slot.value))
        elif slot.status == "scalar":
            setattr(module, slot.attribute, slot.value)
        else:  # pragma: no cover
            raise SnapshotError(f"unknown model-state status: {slot.status!r}")


class ExecutionStateController:
    """Serialize execution of coexisting restored branches on one model."""

    def __init__(self, model: Any):
        self.model = model
        self._controller_id = uuid.uuid4().hex
        self._active_branch_id: str | None = None
        self._epoch = 0

    def restore(self, value: ExecutionSnapshot) -> RestoredExecutionState:
        cache, position, model_state = restore(model=self.model, snapshot=value)
        state = RestoredExecutionState(
            cache, position, model_state, uuid.uuid4().hex, self._controller_id
        )
        self.activate(state)
        return state

    def activate(self, state: RestoredExecutionState) -> None:
        if state._controller_id != self._controller_id:
            raise BranchActivationError("branch belongs to a different controller")
        _apply_model_state(self.model, state.model_state)
        self._epoch += 1
        state._activation_epoch = self._epoch
        self._active_branch_id = state.branch_id

    def assert_active(self, state: RestoredExecutionState) -> None:
        if state._controller_id != self._controller_id:
            raise BranchActivationError("branch belongs to a different controller")
        if self._active_branch_id != state.branch_id or state._activation_epoch != self._epoch:
            raise BranchActivationError(
                "inactive branch; restore or activate it explicitly before execution"
            )

    def forward(self, state: RestoredExecutionState, **kwargs: Any) -> Any:
        self.assert_active(state)
        if "past_key_values" in kwargs:
            raise BranchActivationError("past_key_values is owned by the active branch")
        input_ids = kwargs.get("input_ids")
        if input_ids is not None and (input_ids.ndim != 2 or input_ids.shape[0] != 1):
            raise BranchActivationError("snapshot execution requires batch 1 (A8)")
        return self.model(past_key_values=state.cache, **kwargs)


def iter_snapshot_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every tensor in snapshot/state dataclasses for isolation tests."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            child_prefix = f"{prefix}.{field.name}" if prefix else field.name
            yield from iter_snapshot_tensors(getattr(value, field.name), child_prefix)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from iter_snapshot_tensors(child, f"{prefix}[{index}]")


def tensor_storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    """Return device, storage pointer, and logical pointer (A4)."""
    return str(tensor.device), int(tensor.untyped_storage().data_ptr()), int(tensor.data_ptr())


def snapshot_fingerprint(value: ExecutionSnapshot) -> str:
    """Content fingerprint for local immutability tests, never warmups."""
    tensors: list[dict[str, Any]] = []
    for path, tensor in iter_snapshot_tensors(value):
        raw = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        tensors.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(memoryview(raw.numpy())).hexdigest(),
            }
        )
    metadata = {
        "schema": value.schema_version,
        "cache_type": value.cache_type,
        "position_length": value.position.sequence_length,
        "layer_types": [type(layer).__name__ for layer in value.layers],
        "model_slots": [
            [slot.module_path, slot.attribute, slot.status] for slot in value.model_state
        ],
        "tensors": tensors,
    }
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
