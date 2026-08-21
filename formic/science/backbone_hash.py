"""Canonical streaming content hash for the 851 text-runtime tensors."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from formic.backbone.inventory import CheckpointInventory, InventoryError

ALGORITHM = "formic-text-tensors-sha256-v1"


@dataclass(frozen=True)
class BackboneHash:
    algorithm: str
    sha256: str
    tensor_count: int
    payload_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "algorithm": self.algorithm,
            "sha256": self.sha256,
            "tensor_count": self.tensor_count,
            "payload_bytes": self.payload_bytes,
        }


def canonical_backbone_hash(
    inventory: CheckpointInventory,
    *,
    expected_tensor_count: int = 851,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> BackboneHash:
    """Hash ordered names, metadata and raw stored bytes without GPU transfer."""
    records = tuple(
        sorted(
            (item for item in inventory.records if item.family in ("text", "lm_head")),
            key=lambda item: item.name,
        )
    )
    if len(records) != expected_tensor_count:
        raise InventoryError(
            f"canonical backbone hash expects {expected_tensor_count} tensors, got {len(records)}"
        )
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    header_cache: dict[str, tuple[int, dict]] = {}
    handles: dict[str, BinaryIO] = {}
    aggregate = hashlib.sha256()
    total_bytes = 0
    try:
        for record in records:
            if record.shard not in header_cache:
                header_cache[record.shard] = _header(inventory.path / record.shard)
            data_start, header = header_cache[record.shard]
            meta = header.get(record.name)
            if meta is None:
                raise InventoryError(f"tensor missing from shard header: {record.name}")
            start, stop = (int(offset) for offset in meta["data_offsets"])
            if stop - start != record.nbytes:
                raise InventoryError(f"payload size mismatch for {record.name}")
            handle = handles.setdefault(
                record.shard, (inventory.path / record.shard).open("rb")
            )
            handle.seek(data_start + start)
            tensor_hash = hashlib.sha256()
            remaining = stop - start
            while remaining:
                block = handle.read(min(remaining, chunk_bytes))
                if not block:
                    raise InventoryError(f"truncated payload for {record.name}")
                tensor_hash.update(block)
                remaining -= len(block)
            descriptor = json.dumps(
                {
                    "name": record.name,
                    "dtype": record.dtype,
                    "shape": list(record.shape),
                    "nbytes": record.nbytes,
                    "content_sha256": tensor_hash.hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            aggregate.update(struct.pack("<Q", len(descriptor)))
            aggregate.update(descriptor)
            total_bytes += record.nbytes
    finally:
        for handle in handles.values():
            handle.close()
    return BackboneHash(ALGORITHM, aggregate.hexdigest(), len(records), total_bytes)


def load_reusable_backbone_hash(path: str | Path) -> BackboneHash:
    """Reuse an audit result only when its algorithm and 851 count match."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"algorithm", "sha256", "tensor_count", "payload_bytes"}
    if set(value) != required:
        raise InventoryError("audit backbone hash record has an unexpected schema")
    result = BackboneHash(
        value["algorithm"], value["sha256"], value["tensor_count"], value["payload_bytes"]
    )
    if result.algorithm != ALGORITHM or result.tensor_count != 851:
        raise InventoryError("audit backbone hash is not reusable by SPEC-02")
    if len(result.sha256) != 64:
        raise InventoryError("invalid audit backbone SHA-256")
    return result


def _header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise InventoryError(f"invalid safetensors prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise InventoryError(f"truncated safetensors header: {path}")
    return 8 + header_length, json.loads(raw)
