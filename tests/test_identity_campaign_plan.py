from __future__ import annotations

from dataclasses import replace

from formic.config.loader import load_config
from formic.science.identity.campaign_plan import build_campaign_plan, timing_continuation
from formic.science.identity.prompts import load_frozen_corpus
from formic.science.identity.types import ExecutionMode


def test_final_a40_plan_is_pinned_to_the_balanced_9925_forward_protocol():
    config = load_config("configs/default.yaml")
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    plan = build_campaign_plan(config, corpus)

    assert len(plan.preflight_paths) == 18
    assert plan.total_forwards == 9_925
    assert plan.phase_forwards["trace_inertness"] == 144
    assert plan.phase_forwards["legacy_continuity"] == 3_872
    assert plan.phase_forwards["noise_floor"] == 752
    assert plan.phase_forwards["accumulation_probe_64"] == 2_560
    assert all(
        item.mode is not ExecutionMode.DECODE_RECOMPUTE or item.prompt.length_class != "long"
        for item in plan.calibration_paths
    )
    assert {
        item.segmentation
        for item in plan.calibration_paths
        if item.prompt.length_class == "long" and item.mode is ExecutionMode.PREFILL_SEGMENTED
    } == {"median", "quarters"}
    for length_class in ("short", "medium"):
        modes = [
            item.mode
            for item in plan.calibration_paths
            if item.prompt.length_class == length_class
            and item.mode in (ExecutionMode.DECODE_CACHED, ExecutionMode.DECODE_RECOMPUTE)
        ]
        assert modes == [ExecutionMode.DECODE_RECOMPUTE, ExecutionMode.DECODE_CACHED]


def test_a40_35_gib_resolved_config_hash_is_pinned():
    # The v2 hash validated by the balanced crossover was
    # b0b2ca19b553ea06f41b0cf4f876107bfd843ad08d0ccf5c281123ce3c7965b5; the v3
    # burn-in keys deliberately change the resolved hash (and therefore
    # invalidate every pre-v3 resume).  This pin detects accidental drift.
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            max_memory={**config.backbone.max_memory, "0": "35GiB"},
        ),
    )

    assert config.config_hash() == (
        "742dabcabb9c597c276f32ea448fd3b0fd2535d3d0035957855268c3db07488b"
    )


def test_preflight_continuation_is_the_approved_last_frozen_token_repetition():
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    prompt = next(item for item in corpus.prompts if item.id == "medium_cache_regression")

    assert timing_continuation(prompt, 8) == (prompt.token_ids[-1],) * 8
