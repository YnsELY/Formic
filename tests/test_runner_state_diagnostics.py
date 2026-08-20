"""Weight-free tests for the SPEC-01 runner diagnostic harness."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.step1_runner_state_diagnostics import (
    _observe_top_level,
    _boundary_label,
    _call_log_diff,
    _state_component_diff,
    _state_snapshot,
    _tensor_record,
    _validate_state_trace,
)


class FakeAttentionLayer:
    def __init__(self, value: float):
        self.keys = torch.tensor([[[[value]]]], dtype=torch.bfloat16)
        self.values = torch.tensor([[[[value + 1]]]], dtype=torch.bfloat16)
        self.is_initialized = True


class FakeGdnLayer:
    def __init__(self, value: float):
        self.conv_states = torch.tensor([[[value]]], dtype=torch.bfloat16)
        self.recurrent_states = torch.tensor([[[value + 1]]], dtype=torch.bfloat16)
        self.is_conv_states_initialized = True
        self.is_recurrent_states_initialized = True
        self.has_previous_state = True


class FakeModel:
    def named_modules(self):
        return [("", self)]


def test_boundary_labels_distinguish_prefill_from_consumed_tokens():
    assert _boundary_label(0) == "prefill"
    assert _boundary_label(1) == "after_forced_0"
    assert _boundary_label(15) == "after_forced_14"


def test_tensor_record_hashes_raw_dtype_and_content():
    left = _tensor_record(torch.tensor([1, 2], dtype=torch.int64))
    right = _tensor_record(torch.tensor([1, 2], dtype=torch.int64))
    changed = _tensor_record(torch.tensor([1, 3], dtype=torch.int64))
    assert left["dtype"] == "int64"
    assert left["sha256"] == right["sha256"]
    assert left["sha256"] != changed["sha256"]


def test_call_diff_reports_first_nested_field():
    left = [{"arguments": {"input_ids": {"status": "absent"}}}]
    right = [{"arguments": {"input_ids": {"status": "present"}}}]
    result = _call_log_diff(left, right)
    assert not result["exact"]
    assert result["first_difference"]["field"] == "calls[0].arguments.input_ids.status"


def test_state_diff_identifies_boundary_layer_and_component():
    model = FakeModel()
    left_cache = SimpleNamespace(layers=[FakeAttentionLayer(1), FakeGdnLayer(2)])
    right_cache = SimpleNamespace(layers=[FakeAttentionLayer(1), FakeGdnLayer(2)])
    right_cache.layers[1].recurrent_states.add_(1)
    left = [_state_snapshot(model, left_cache, 0)]
    right = [_state_snapshot(model, right_cache, 0)]
    result = _state_component_diff(left, right)
    first = result["first_difference"]
    assert first["boundary"] == "prefill"
    assert first["layer"] == 1
    assert first["component"] == "recurrent_states"


def test_state_diff_is_exact_for_equal_snapshots():
    model = FakeModel()
    cache = SimpleNamespace(layers=[FakeAttentionLayer(1), FakeGdnLayer(2)])
    left = [_state_snapshot(model, cache, 0)]
    right = [_state_snapshot(model, cache, 0)]
    assert _state_component_diff(left, right)["exact"]


def test_state_diff_detects_layer_count_and_flag_changes():
    model = FakeModel()
    left_cache = SimpleNamespace(layers=[FakeAttentionLayer(1), FakeGdnLayer(2)])
    right_cache = SimpleNamespace(layers=[FakeAttentionLayer(1)])
    left = [_state_snapshot(model, left_cache, 0)]
    right = [_state_snapshot(model, right_cache, 0)]
    assert _state_component_diff(left, right)["first_difference"]["component"] == "layer_count"

    right_cache.layers.append(FakeGdnLayer(2))
    right_cache.layers[1].has_previous_state = False
    right = [_state_snapshot(model, right_cache, 0)]
    assert (
        _state_component_diff(left, right)["first_difference"]["component"]
        == "metadata.has_previous_state"
    )


def test_top_level_state_observer_returns_none_and_is_removed():
    class Output:
        past_key_values = SimpleNamespace(layers=[])

    class Model(torch.nn.Module):
        def forward(self, input_ids=None, use_cache=None):
            return Output()

    model = Model()
    pre_before = len(model._forward_pre_hooks)
    post_before = len(model._forward_hooks)
    with _observe_top_level(model, capture_state=True) as observer:
        output = model(input_ids=torch.tensor([[1]]), use_cache=True)
        assert isinstance(output, Output)
    assert len(observer.calls) == 1
    assert len(observer.states) == 1
    assert len(model._forward_pre_hooks) == pre_before
    assert len(model._forward_hooks) == post_before


def test_top_level_observer_is_removed_after_exception():
    class Model(torch.nn.Module):
        def forward(self, input_ids=None):
            raise RuntimeError("boom")

    model = Model()
    try:
        with _observe_top_level(model, capture_state=False):
            model(input_ids=torch.tensor([[1]]))
    except RuntimeError:
        pass
    assert len(model._forward_pre_hooks) == 0
    assert len(model._forward_hooks) == 0


def test_state_trace_completeness_rejects_partial_cache():
    model = FakeModel()
    states = [_state_snapshot(model, SimpleNamespace(layers=[]), index) for index in range(16)]
    try:
        _validate_state_trace(states, prompt_length=4)
    except RuntimeError as error:
        assert "expected 64 cache layers" in str(error)
    else:
        raise AssertionError("partial cache was accepted")
