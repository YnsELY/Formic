from __future__ import annotations

import gc
import json
import weakref

import torch

from formic.backbone.groups import HybridGroupView
from formic.science.identity.executor import Endpoint
from formic.science.identity import schedule_diagnostic
from tests.toy_qwen import toy_model


def _endpoints():
    model = toy_model(seed=47)
    view = HybridGroupView.from_text_config(model.config)
    return (
        Endpoint("reference", model, view, False),
        Endpoint("runner", model, view, True),
    )


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_alternating_warmup_disables_autograd_releases_outputs_and_records_order(monkeypatch):
    left, right = _endpoints()
    original = schedule_diagnostic._call_endpoint
    output_refs = []
    grad_observations = []

    def observed_call(endpoint, input_ids, cache):
        grad_observations.append((endpoint.name, torch.is_grad_enabled()))
        output = original(endpoint, input_ids, cache)
        output_refs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(schedule_diagnostic, "_call_endpoint", observed_call)
    events = []
    memory_labels = []
    result = schedule_diagnostic.run_schedule_pair(
        "alternating",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=False,
        event_observer=events.append,
        memory_observer=memory_labels.append,
    )
    gc.collect()

    assert result is None
    assert grad_observations == [
        ("reference", False), ("runner", False),
        ("reference", False), ("runner", False),
        ("reference", False), ("runner", False),
    ]
    assert all(reference() is None for reference in output_refs)
    assert [
        (event["side"], event["step"])
        for event in events
        if event["event"] == "after_endpoint"
    ] == [
        ("left", 0), ("right", 0),
        ("left", 1), ("right", 1),
        ("left", 2), ("right", 2),
    ]
    left_cache_ids = {
        event["cache_object_id"] for event in events if event["side"] == "left"
    }
    right_cache_ids = {
        event["cache_object_id"] for event in events if event["side"] == "right"
    }
    assert len(left_cache_ids) == len(right_cache_ids) == 1
    assert left_cache_ids.isdisjoint(right_cache_ids)
    assert memory_labels == [
        "before_cache_creation",
        "after_left_step_0", "after_left_step_0_output_deleted",
        "after_right_step_0", "after_right_step_0_output_deleted",
        "after_left_step_1", "after_left_step_1_output_deleted",
        "after_right_step_1", "after_right_step_1_output_deleted",
        "after_left_step_2", "after_left_step_2_output_deleted",
        "after_right_step_2", "after_right_step_2_output_deleted",
        "after_warmup",
    ]


def test_measured_alternating_result_has_independent_caches_and_no_tensors():
    left, right = _endpoints()
    result = schedule_diagnostic.run_schedule_pair(
        "alternating",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=True,
    )

    assert result is not None
    assert result["autograd_disabled_all_forwards"] is True
    assert result["cache_independence"] == {
        "cache_objects_distinct": True,
        "cache_storage_disjoint": True,
        "fresh_cache_pair_constructed_for_call": True,
    }
    assert not _contains_tensor(result)
    assert all(step["left"]["device"] == "cpu" for step in result["steps"])
    assert all(step["right"]["device"] == "cpu" for step in result["steps"])
    json.dumps(result)


def test_sequential_calendar_runs_whole_left_path_before_right_path():
    left, right = _endpoints()
    result = schedule_diagnostic.run_schedule_pair(
        "sequential",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=True,
    )

    assert result is not None
    assert result["forward_order"] == [
        {"side": "left", "step": 0},
        {"side": "left", "step": 1},
        {"side": "left", "step": 2},
        {"side": "right", "step": 0},
        {"side": "right", "step": 1},
        {"side": "right", "step": 2},
    ]
