"""Incremental CUDA memory measurements for SPEC-02 campaign controls."""

from __future__ import annotations

import gc
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from formic.science.identity.artifacts import atomic_write_json


def cuda_memory_measurement(label: str, model: Any | None = None) -> dict[str, Any]:
    """Measure allocator headroom and attribute live model-owned storages."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA memory measurement requires CUDA")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    parameter_storages = _storages(model.parameters()) if model is not None else {}
    buffer_storages = _storages(model.buffers()) if model is not None else {}
    parameter_bytes = sum(parameter_storages.values())
    buffer_bytes = sum(
        size for pointer, size in buffer_storages.items() if pointer not in parameter_storages
    )
    allocated = int(torch.cuda.memory_allocated())
    return {
        "label": label,
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
        "allocated_bytes": allocated,
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "model_parameter_storage_bytes": parameter_bytes,
        "model_buffer_storage_bytes": buffer_bytes,
        "allocated_outside_model_storage_bytes": allocated - parameter_bytes - buffer_bytes,
    }


def live_cuda_tensor_summary(model: Any) -> dict[str, Any]:
    """Summarise Python-reachable CUDA tensor storages after a control run."""
    parameter_pointers = set(_storages(model.parameters()))
    buffer_pointers = set(_storages(model.buffers()))
    storages: dict[int, tuple[int, str, tuple[int, ...]]] = {}
    for value in gc.get_objects():
        try:
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                continue
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            if pointer not in storages:
                storages[pointer] = (int(storage.nbytes()), str(value.dtype), tuple(value.shape))
        except (ReferenceError, RuntimeError):
            continue
    categories = Counter()
    shapes = Counter()
    for pointer, (size, dtype, shape) in storages.items():
        if pointer in parameter_pointers:
            category = "model_parameter"
        elif pointer in buffer_pointers:
            category = "model_buffer"
        else:
            category = "other_python_reachable"
        categories[category] += size
        if category == "other_python_reachable":
            shapes[(shape, dtype)] += size
    return {
        "storage_bytes_by_category": dict(categories),
        "largest_other_storages": [
            {"shape": list(shape), "dtype": dtype, "bytes": size}
            for (shape, dtype), size in shapes.most_common(20)
        ],
    }


class IncrementalMemoryWriter:
    """Persist every memory observation atomically as soon as it is recorded."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.measurements: list[dict[str, Any]] = []

    def record(self, label: str, model: Any | None = None) -> dict[str, Any]:
        measurement = cuda_memory_measurement(label, model)
        self.measurements.append(measurement)
        atomic_write_json(
            self.path,
            {"schema_version": 1, "measurements": self.measurements},
        )
        return measurement

    def write_live_summary(self, model: Any) -> None:
        atomic_write_json(self.path.with_name("live_tensors.json"), live_cuda_tensor_summary(model))


def _storages(values: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            continue
        storage = value.untyped_storage()
        result[int(storage.data_ptr())] = int(storage.nbytes())
    return result
