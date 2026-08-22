from __future__ import annotations

from formic.config.loader import load_config
from formic.science.identity.campaign_plan import build_campaign_plan, timing_continuation
from formic.science.identity.prompts import load_frozen_corpus
from formic.science.identity.types import ExecutionMode


def test_final_a40_plan_is_pinned_to_the_approved_4139_forward_protocol():
    config = load_config("configs/default.yaml")
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    plan = build_campaign_plan(config, corpus)

    assert len(plan.preflight_paths) == 18
    assert plan.total_forwards == 4_139
    assert plan.phase_forwards["accumulation_probe_64"] == 1_280
    assert all(
        item.mode is not ExecutionMode.DECODE_RECOMPUTE or item.prompt.length_class != "long"
        for item in plan.calibration_paths
    )
    assert {
        item.segmentation
        for item in plan.calibration_paths
        if item.prompt.length_class == "long" and item.mode is ExecutionMode.PREFILL_SEGMENTED
    } == {"median", "quarters"}


def test_preflight_continuation_is_the_approved_last_frozen_token_repetition():
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    prompt = next(item for item in corpus.prompts if item.id == "medium_cache_regression")

    assert timing_continuation(prompt, 8) == (prompt.token_ids[-1],) * 8
