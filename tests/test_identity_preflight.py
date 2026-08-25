from __future__ import annotations

import torch

from formic.config.loader import load_config
from formic.science.identity.campaign_plan import build_campaign_plan
from formic.science.identity.preflight import (
    PathTiming,
    _estimate_from_timings,
    release_cuda_working_set,
)
from formic.science.identity.prompts import load_frozen_corpus


def test_preflight_estimate_is_schema_valid_and_informational_for_every_phase():
    config = load_config("configs/default.yaml")
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    plan = build_campaign_plan(config, corpus)
    timings = [PathTiming(path, 1.0, (1.2, 1.3)) for path in plan.preflight_paths]

    estimate = _estimate_from_timings(
        timings,
        model_load_seconds=240.0,
        preflight_elapsed_seconds=120.0,
        transfer_bytes_per_second=8 * 2**30,
    )

    estimate.validate()
    assert estimate.model_processes == 1
    assert estimate.preflight_forwards == 333
    assert [item.name for item in estimate.phases] == [
        "trace_inertness",
        "legacy_continuity",
        "noise_floor",
        "snapshot_restore",
        "reference_continuations",
        "short",
        "medium",
        "long",
        "accumulation_probe_64",
    ]


def test_release_cuda_working_set_flushes_inactive_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    release_cuda_working_set()

    assert calls == ["synchronize", "empty_cache"]


def test_distinct_canonical_reference_timing_is_not_replaced_by_candidate_time():
    config = load_config("configs/default.yaml")
    corpus = load_frozen_corpus("configs/reference_prompts.yaml")
    plan = build_campaign_plan(config, corpus)
    timings = []
    for path in plan.preflight_paths:
        reference = None
        if path.mode.value == "prefill_segmented":
            reference = (4.0, 5.0)
        timings.append(PathTiming(path, 1.0, (1.0, 1.0), 4.0, reference))

    estimate = _estimate_from_timings(
        timings,
        model_load_seconds=1.0,
        preflight_elapsed_seconds=1.0,
        transfer_bytes_per_second=8 * 2**30,
    )

    short = next(item for item in estimate.phases if item.name == "short")
    assert short.estimated_seconds > 18 * 4 * 5.0
