"""Strict tensor inventory (A12): permissive loading must be impossible.

Weight-free: only safetensors headers are read.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from formic.backbone import constants as C
from formic.backbone.inventory import (
    CheckpointInventory,
    InventoryError,
    classify,
    remap_key,
)

CHECKPOINT = Path("/workspace/Qwen3.8-27B")


@pytest.fixture(scope="module")
def inventory() -> CheckpointInventory:
    return CheckpointInventory.from_checkpoint(CHECKPOINT)


def test_inventory_matches_audit_totals(inventory: CheckpointInventory):
    assert len(inventory.records) == C.TOTAL_TENSORS == 1_199
    assert inventory.params() == C.TOTAL_STORED_PARAMS == 27_781_427_952
    assert inventory.total_bytes() == C.TOTAL_PAYLOAD_BYTES == 55_562_855_904
    assert len(inventory.shards()) == C.NUM_SHARDS == 18


def test_all_tensors_are_bf16(inventory: CheckpointInventory):
    assert {record.dtype for record in inventory.records} == {"bfloat16"}


def test_family_counts_and_params(inventory: CheckpointInventory):
    assert len(inventory.by_family("vision")) == C.VISION_TENSORS == 333
    assert len(inventory.by_family("mtp")) == C.MTP_TENSORS == 15
    assert len(inventory.by_family("lm_head")) == 1
    assert inventory.params("mtp") == C.MTP_PARAMS == 424_699_392
    assert inventory.params("vision") == C.VISION_PARAMS == 460_730_096
    assert inventory.params("lm_head") == C.LM_HEAD_PARAMS


def test_audit_validation_passes(inventory: CheckpointInventory):
    inventory.validate_against_audit()  # raises on any divergence


def test_expected_tensors_text_only_excludes_vision_and_mtp(inventory: CheckpointInventory):
    expected = inventory.expected_model_tensors("text_only")
    assert len(expected) == 851  # 850 text + lm_head
    assert not any(name.startswith("model.visual") for name in expected)
    assert not any(name.startswith("mtp") for name in expected)
    assert "model.embed_tokens.weight" in expected
    assert "lm_head.weight" in expected
    assert expected["lm_head.weight"] == (C.VOCAB_SIZE, C.HIDDEN_SIZE)


def test_audit_multimodal_inventory_matches_transformers_accounting(inventory: CheckpointInventory):
    """The audited 'loaded by Transformers' figure is reproduced exactly."""
    expected = inventory.expected_model_tensors("audit_multimodal")
    total = 0
    for shape in expected.values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    assert total == C.PARAMS_LOADED_BY_TRANSFORMERS == 27_356_728_560
    assert len(expected) == C.TOTAL_TENSORS - C.MTP_TENSORS


def test_exclusions_are_declared_not_silent(inventory: CheckpointInventory):
    assert inventory.declared_exclusions("text_only") == {"mtp": 15, "vision": 333}
    assert inventory.declared_exclusions("audit_multimodal") == {"mtp": 15}


def test_key_mapping_is_a_pure_prefix_rename(inventory: CheckpointInventory):
    mapping = inventory.key_mapping("text_only")
    assert mapping == {r"^model\.language_model\.": "model."}
    assert inventory.key_mapping("audit_multimodal") == {}


def test_text_only_key_mapping_is_a_strict_metadata_preserving_bijection(
    inventory: CheckpointInventory,
):
    name_map = inventory.text_only_name_mapping()
    source_records = {
        record.name: record
        for record in inventory.records
        if record.family in {"text", "lm_head"}
    }
    expected = inventory.expected_model_tensors("text_only")

    assert set(name_map) == set(source_records)
    assert len(name_map) == len(set(name_map.values())) == 851
    assert set(name_map.values()) == set(expected)

    inverse = {target: source for source, target in name_map.items()}
    assert all(inverse[target] == source for source, target in name_map.items())

    regex, replacement = next(iter(inventory.key_mapping("text_only").items()))
    for source, target in name_map.items():
        record = source_records[source]
        assert re.sub(regex, replacement, source) == target
        assert expected[target] == record.shape
        assert record.dtype == C.CHECKPOINT_DTYPE
        assert record.num_params == _numel(record.shape)

    assert inventory.text_only_mapping_report() == {
        "source_tensors": 851,
        "target_tensors": 851,
        "renamed_tensors": 850,
        "unchanged_tensors": 1,
        "injective": True,
        "surjective_onto_expected": True,
        "roundtrip": True,
        "metadata_preserved": True,
        "regex_matches_name_map": True,
    }


def test_remap_key_roundtrip():
    src = "model.language_model.layers.7.self_attn.q_proj.weight"
    assert remap_key(src, "text_only") == "model.layers.7.self_attn.q_proj.weight"
    assert remap_key(src, "audit_multimodal") == src
    assert remap_key("lm_head.weight", "text_only") == "lm_head.weight"
    assert remap_key("model.visual.blocks.0.norm1.weight", "text_only").startswith("model.visual")


def test_classification_is_exhaustive_and_strict(inventory: CheckpointInventory):
    for record in inventory.records:
        assert record.family in {"text", "lm_head", "vision", "mtp"}
    with pytest.raises(InventoryError):
        classify("some.unknown.tensor")


def test_layer_mixer_families_follow_the_hybrid_pattern(inventory: CheckpointInventory):
    gdn_layers, attention_layers = set(), set()
    for record in inventory.by_family("text"):
        suffix = record.name[len(C.CKPT_TEXT_PREFIX) :]
        if not suffix.startswith("layers."):
            continue
        parts = suffix.split(".")
        index = int(parts[1])
        if parts[2] == "linear_attn":
            gdn_layers.add(index)
        elif parts[2] == "self_attn":
            attention_layers.add(index)
    assert sorted(attention_layers) == list(C.ATTENTION_LAYER_INDICES)
    assert len(gdn_layers) == C.NUM_GDN_LAYERS
    assert gdn_layers & attention_layers == set()


def test_missing_checkpoint_is_fatal(tmp_path: Path):
    with pytest.raises(InventoryError):
        CheckpointInventory.from_checkpoint(tmp_path)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
