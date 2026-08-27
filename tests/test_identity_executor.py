from __future__ import annotations

from formic.backbone.groups import HybridGroupView
from formic.science.identity.executor import (
    Endpoint,
    AlignedCasePayload,
    expected_shapes,
    reference_shapes_for_candidate,
    run_aligned_pair,
    run_cross_path_pair,
    run_greedy_pair,
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


def test_cross_path_cached_decode_uses_recompute_reference_and_cpu_evidence():
    reference_model = toy_model(seed=45)
    candidate_model = toy_model(seed=45)
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
    result = run_cross_path_pair(
        reference,
        candidate,
        prompt_token_ids=(1, 2, 3, 4),
        candidate_mode=ExecutionMode.DECODE_CACHED,
        length_class="short",
        forced_token_ids=(5, 6, 7),
        capture=True,
    )

    assert isinstance(result.payload, AlignedCasePayload)
    assert [frame.shape.key for frame in result.payload.reference.frames] == [
        "b1-i4-c0",
        "b1-i5-c0",
        "b1-i6-c0",
    ]
    assert [frame.shape.key for frame in result.payload.candidate.frames] == [
        "b1-i4-c0",
        "b1-i1-c4",
        "b1-i1-c5",
    ]
    assert all(
        frame.trace.logits.device.type == "cpu"
        for path in (result.payload.reference, result.payload.candidate)
        for frame in path.frames
    )
    assert {
        item.point.value
        for comparison in result.payload.comparisons
        for item in comparison.not_applicable
    } == {"gdn_state", "attention_kv"}


def test_cross_path_logits_only_override_captures_no_boundary_state():
    """The 64-frame probe is logits-only end to end, not just at serialisation."""
    reference_model = toy_model(seed=45)
    candidate_model = toy_model(seed=45)
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
    from formic.science.identity.types import CaptureProfile

    result = run_cross_path_pair(
        reference,
        candidate,
        prompt_token_ids=(1, 2, 3, 4),
        candidate_mode=ExecutionMode.DECODE_CACHED,
        length_class="short",
        forced_token_ids=(5, 6, 7),
        capture=True,
        capture_profile=CaptureProfile.LOGITS_ONLY,
    )

    assert isinstance(result.payload, AlignedCasePayload)
    points = {
        item.location.point.value
        for comparison in result.payload.comparisons
        for item in comparison.measurements
    }
    assert points == {"logits"}
    for path in (result.payload.reference, result.payload.candidate):
        for frame in path.frames:
            assert frame.trace.boundaries == ()
            assert frame.trace.final_state is None
    # One retained tensor per frame (the logits), nothing else.
    assert result.captured_state_tensors == len(
        result.payload.reference.frames
    ) + len(result.payload.candidate.frames)


def test_reference_shapes_for_segmented_prefill_are_full_prefixes():
    assert [
        shape.key
        for shape in reference_shapes_for_candidate(
            prompt_length=9,
            candidate_mode=ExecutionMode.PREFILL_SEGMENTED,
            segmentation="median",
            decode_steps=3,
            length_class="short",
        )
    ] == ["b1-i4-c0", "b1-i9-c0"]


def test_greedy_pair_forces_reference_tokens_without_a_separate_generation_pass():
    reference_model = toy_model(seed=44)
    candidate_model = toy_model(seed=44)
    reference = Endpoint(
        "reference", reference_model, HybridGroupView.from_text_config(reference_model.config), False
    )
    candidate = Endpoint(
        "runner", candidate_model, HybridGroupView.from_text_config(candidate_model.config), True
    )
    result = run_greedy_pair(
        reference,
        candidate,
        prompt_token_ids=(1, 2, 3, 4),
        length_class="short",
        decode_steps=3,
        capture=True,
    )
    assert isinstance(result.payload, AlignedCasePayload)
    assert len(result.payload.comparisons) == 3
    assert all(
        (item.metric.tensor if hasattr(item.metric, "tensor") else item.metric).exact
        for comparison in result.payload.comparisons
        for item in comparison.measurements
    )
