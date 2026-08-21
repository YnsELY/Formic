"""Weight-free checks for strict Hugging Face loading diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from formic.backbone.inventory import AUDITED_INVENTORY_MANIFEST, CheckpointInventory
from formic.backbone.loader import BackboneLoadError, validate_hf_loading_info

@pytest.fixture(scope="module")
def inventory() -> CheckpointInventory:
    return CheckpointInventory.from_audited_manifest(AUDITED_INVENTORY_MANIFEST)


def test_empty_hf_loading_info_is_strict(inventory: CheckpointInventory):
    report = validate_hf_loading_info({}, inventory)
    assert report == {
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_keys": 0,
        "error_messages": 0,
        "reported_declared_exclusions": 0,
    }


def test_declared_exclusions_may_be_reported(inventory: CheckpointInventory):
    report = validate_hf_loading_info(
        {"unexpected_keys": ["mtp.fc.weight", "model.visual.patch_embed.proj.weight"]},
        inventory,
    )
    assert report["reported_declared_exclusions"] == 2


@pytest.mark.parametrize(
    "loading_info",
    (
        {"missing_keys": ["model.layers.0.input_layernorm.weight"]},
        {"unexpected_keys": ["model.language_model.not_a_real_weight"]},
        {"mismatched_keys": [("model.embed_tokens.weight", (1,), (2,))]},
        {"error_msgs": ["load failed"]},
    ),
)
def test_any_undeclared_loading_problem_is_fatal(
    inventory: CheckpointInventory, loading_info: dict
):
    with pytest.raises(BackboneLoadError, match="loading_info is not strict"):
        validate_hf_loading_info(loading_info, inventory)
