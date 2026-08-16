"""Hybrid group view: mapping groups <-> layers must be exact (step-1 checklist).

These tests are weight-free: the view is built from ``layer_types`` alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formic.backbone import constants as C
from formic.backbone.groups import (
    BOUNDARY_NAMES,
    GroupStructureError,
    HybridGroupView,
)

CHECKPOINT = Path("/workspace/Qwen3.8-27B")


@pytest.fixture(scope="module")
def view() -> HybridGroupView:
    raw = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    return HybridGroupView.from_checkpoint_config(raw)


def test_view_is_built_from_the_real_checkpoint_config(view: HybridGroupView):
    assert len(view.layer_types) == C.NUM_LAYERS
    assert len(view.groups) == C.NUM_GROUPS
    assert len(view.layers) == C.NUM_LAYERS


def test_attention_layers_are_at_the_audited_indices(view: HybridGroupView):
    assert view.attention_layer_indices() == C.ATTENTION_LAYER_INDICES
    assert len(view.gdn_layer_indices()) == C.NUM_GDN_LAYERS


def test_each_group_is_three_gdn_plus_one_attention(view: HybridGroupView):
    for group in view.groups:
        assert len(group.layer_indices) == C.GROUP_SIZE
        assert group.layer_types == C.GROUP_PATTERN
        assert group.attention_layer == group.last_layer
        assert len(group.gdn_layers) == 3
        assert all(view.layer(i).is_gdn for i in group.gdn_layers)
        assert view.layer(group.attention_layer).is_attention


def test_groups_partition_the_stack_without_overlap(view: HybridGroupView):
    covered: list[int] = []
    for group in view.groups:
        covered.extend(group.layer_indices)
    assert covered == list(range(C.NUM_LAYERS))


def test_layer_specs_are_consistent(view: HybridGroupView):
    for spec in view.layers:
        assert spec.group_index == C.group_index_of_layer(spec.index)
        assert spec.position_in_group == spec.index % C.GROUP_SIZE
        assert spec.is_attention != spec.is_gdn
        assert spec.mixer_attr == ("self_attn" if spec.is_attention else "linear_attn")


def test_seventeen_boundaries_named_and_ordered(view: HybridGroupView):
    assert len(view.boundaries) == C.NUM_BOUNDARIES == 17
    assert tuple(b.name for b in view.boundaries) == BOUNDARY_NAMES
    assert BOUNDARY_NAMES[0] == "PRE_G1"
    assert BOUNDARY_NAMES[-1] == "POST_G16"

    first, last = view.boundary("PRE_G1"), view.boundary("POST_G16")
    assert first.before_layer == 0 and first.after_layer is None and first.is_pre_stack
    assert last.after_layer == 63 and last.before_layer is None and last.is_post_stack

    middle = view.boundary("G4_G5")
    assert middle.after_layer == 15 and middle.before_layer == 16
    assert middle.upstream_group == 4 and middle.downstream_group == 5


def test_boundaries_sit_exactly_between_groups(view: HybridGroupView):
    for boundary in view.boundaries:
        if boundary.after_layer is not None:
            assert boundary.after_layer in C.ATTENTION_LAYER_INDICES  # groups end on attention
        if boundary.before_layer is not None:
            assert boundary.before_layer % C.GROUP_SIZE == 0  # groups start on a GDN


def test_group_entry_exit_boundary_names(view: HybridGroupView):
    assert view.group(1).entry_boundary == "PRE_G1"
    assert view.group(1).exit_boundary == "G1_G2"
    assert view.group(16).entry_boundary == "G15_G16"
    assert view.group(16).exit_boundary == "POST_G16"


def test_contiguous_prefixes_match_cape_r_routes(view: HybridGroupView):
    """L0/L1/L2 of CAPE-R are contiguous prefixes ending on group borders."""
    assert view.prefix_layers(8) == tuple(range(32))
    assert view.prefix_layers(12) == tuple(range(48))
    assert view.prefix_layers(16) == tuple(range(64))
    for through in (8, 12, 16):
        assert view.prefix_layers(through)[-1] in C.ATTENTION_LAYER_INDICES


def test_wrong_pattern_is_rejected():
    types = list(C.expected_layer_types())
    types[3] = C.LINEAR_ATTENTION_TYPE  # break the 3+1 pattern
    with pytest.raises(GroupStructureError):
        HybridGroupView(types)


def test_wrong_layer_count_is_rejected():
    with pytest.raises(GroupStructureError):
        HybridGroupView(list(C.expected_layer_types())[:60])


def test_unknown_boundary_name_is_rejected(view: HybridGroupView):
    with pytest.raises(ValueError):
        view.boundary("G17_G18")


def test_describe_is_serialisable(view: HybridGroupView):
    payload = json.dumps(view.describe())
    assert '"num_groups": 16' in payload
