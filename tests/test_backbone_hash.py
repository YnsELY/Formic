from __future__ import annotations

import torch
import pytest
from safetensors.torch import save_file

from formic.backbone.inventory import CheckpointInventory, InventoryError, TensorRecord
from formic.science.backbone_hash import canonical_backbone_hash


def _inventory(root, *, changed=False):
    first = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    if changed:
        first[0, 0] = 9
    second = torch.ones(2, 3, dtype=torch.bfloat16)
    save_file(
        {
            "model.language_model.embed_tokens.weight": first,
            "lm_head.weight": second,
        },
        root / "model.safetensors",
    )
    records = (
        TensorRecord(
            "lm_head.weight", (2, 3), "bfloat16", 6, "model.safetensors", "lm_head"
        ),
        TensorRecord(
            "model.language_model.embed_tokens.weight",
            (2, 3),
            "bfloat16",
            6,
            "model.safetensors",
            "text",
        ),
    )
    return CheckpointInventory(root, records)


def test_canonical_backbone_hash_is_content_sensitive_and_streamed(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = canonical_backbone_hash(
        _inventory(first_root), expected_tensor_count=2, chunk_bytes=3
    )
    same = canonical_backbone_hash(
        _inventory(second_root), expected_tensor_count=2, chunk_bytes=5
    )
    assert first.sha256 == same.sha256
    assert first.tensor_count == 2
    assert first.payload_bytes == 24

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed = canonical_backbone_hash(
        _inventory(changed_root, changed=True), expected_tensor_count=2
    )
    assert changed.sha256 != first.sha256


def test_canonical_hash_rejects_wrong_tensor_count(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    with pytest.raises(InventoryError, match="expects 851"):
        canonical_backbone_hash(_inventory(root))
