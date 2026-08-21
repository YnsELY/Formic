from __future__ import annotations

from formic.backbone.groups import HybridGroupView
from formic.science.identity.executor import (
    Endpoint,
    AlignedCasePayload,
    expected_shapes,
    run_aligned_pair,
)
from formic.science.identity.types import ExecutionMode
from tests.toy_qwen import toy_model
import pytest


def test_expected_shapes_cover_all_four_execution_paths():
    assert [shape.key for shape in expected_shapes(
        prompt_length=8,
        mode=ExecutionMode.PREFILL_FULL,
        segmentation=None,
        decode_steps=3,
    )] == ["b1-i8-c0"]
    assert [shape.key for shape in expected_shapes(
        prompt_length=9,
        mode=ExecutionMode.PREFILL_SEGMENTED,
        segmentation="median",
        decode_steps=3,
    )] == ["b1-i4-c0", "b1-i5-c4"]
    assert [shape.key for shape in expected_shapes(
        prompt_length=8,
        mode=ExecutionMode.DECODE_CACHED,
        segmentation=None,
        decode_steps=3,
    )] == ["b1-i8-c0", "b1-i1-c8", "b1-i1-c9"]
    assert [shape.key for shape in expected_shapes(
        prompt_length=8,
        mode=ExecutionMode.DECODE_RECOMPUTE,
        segmentation=None,
        decode_steps=3,
    )] == ["b1-i8-c0", "b1-i9-c0", "b1-i10-c0"]


def test_aligned_toy_runner_and_reference_compare_all_points_exactly():
    reference_model = toy_model(seed=41)
    candidate_model = toy_model(seed=41)
    reference = Endpoint(
        "reference",
        reference_model,
        HybridGroupView.from_text_config(reference_model.config),
        False,
    )
    candidate = Endpoint(
        "runner",
        candidate_model,
        HybridGroupView.from_text_config(candidate_model.config),
        True,
    )
    result = run_aligned_pair(
        reference,
        candidate,
        prompt_token_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        mode=ExecutionMode.DECODE_CACHED,
        length_class="short",
        forced_token_ids=(9, 10, 11),
        capture=True,
    )
    assert isinstance(result.payload, AlignedCasePayload)
    assert len(result.payload.comparisons) == 3
    measurements = [
        item
        for comparison in result.payload.comparisons
        for item in comparison.measurements
    ]
    points = {item.location.point.value for item in measurements}
    assert points == {
        "logits", "hidden_state", "gdn_state", "attention_kv", "model_state"
    }
    assert all(
        (item.metric.tensor if hasattr(item.metric, "tensor") else item.metric).exact
        for item in measurements
    )
    assert result.captured_state_tensors > 0


def test_long_recompute_decode_is_rejected_by_campaign_scope():
    model = toy_model(seed=43)
    endpoint = Endpoint(
        "reference",
        model,
        HybridGroupView.from_text_config(model.config),
        False,
    )
    with pytest.raises(ValueError, match="disabled for long"):
        from formic.science.identity.executor import execute_path

        execute_path(
            endpoint,
            prompt_token_ids=(1, 2, 3),
            mode=ExecutionMode.DECODE_RECOMPUTE,
            length_class="long",
            forced_token_ids=(4, 5),
            capture=False,
        )
