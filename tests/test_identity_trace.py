from __future__ import annotations

from types import SimpleNamespace

import torch

from formic.backbone.groups import BOUNDARY_NAMES, HybridGroupView
from formic.backbone.runner import identity_forward
from formic.science.identity.trace import IdentityTraceCollector
from formic.science.identity.types import CaptureProfile
from formic.state.snapshot import PositionState
from tests.toy_qwen import cache_tensors, toy_model


def _cache(config):
    from transformers.cache_utils import DynamicCache

    return DynamicCache(config=config)


def _handle(model):
    return SimpleNamespace(model=model)


def test_trace_on_is_bit_inert_and_captures_each_group_once():
    model = toy_model(seed=23)
    view = HybridGroupView.from_text_config(model.config)
    prompt = torch.tensor([[3, 8, 5, 12]], dtype=torch.long)

    plain_cache = _cache(model.config)
    plain = identity_forward(
        _handle(model),
        input_ids=prompt,
        past_key_values=plain_cache,
        use_cache=True,
    )

    traced_cache = _cache(model.config)
    collector = IdentityTraceCollector(
        model=model,
        view=view,
        cache=traced_cache,
        capture_profile=CaptureProfile.FULL_BOUNDARIES,
    )
    traced = identity_forward(
        _handle(model),
        trace_collector=collector,
        input_ids=prompt,
        past_key_values=traced_cache,
        use_cache=True,
    )

    assert torch.equal(plain.logits, traced.logits)
    assert all(
        torch.equal(cache_tensors(plain_cache)[name], cache_tensors(traced_cache)[name])
        for name in cache_tensors(plain_cache)
    )
    assert collector.last_trace is not None
    trace = collector.last_trace
    assert tuple(item.name for item in trace.boundaries) == BOUNDARY_NAMES
    assert trace.boundaries[0].completed_group is None
    assert trace.boundaries[0].cache_applicability == "not_applicable"
    assert trace.boundaries[0].cache_layers == ()
    assert all(len(item.cache_layers) == 4 for item in trace.boundaries[1:])
    captured_indices = tuple(
        layer.layer_index for item in trace.boundaries for layer in item.cache_layers
    )
    assert captured_indices == tuple(range(64))


def test_recompute_marks_cache_state_not_applicable():
    model = toy_model(seed=29)
    collector = IdentityTraceCollector(
        model=model,
        view=HybridGroupView.from_text_config(model.config),
        cache=None,
        capture_profile=CaptureProfile.FULL_BOUNDARIES,
    )
    identity_forward(
        _handle(model),
        trace_collector=collector,
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        use_cache=False,
    )
    assert collector.last_trace is not None
    assert all(
        item.cache_applicability == "not_applicable" and not item.cache_layers
        for item in collector.last_trace.boundaries
    )


def test_long_profile_has_no_boundary_hooks_and_one_final_snapshot():
    model = toy_model(seed=31)
    cache = _cache(model.config)
    collector = IdentityTraceCollector(
        model=model,
        view=HybridGroupView.from_text_config(model.config),
        cache=cache,
        capture_profile=CaptureProfile.FINAL_STATE_ONLY,
    )
    identity_forward(
        _handle(model),
        trace_collector=collector,
        position_state=PositionState(sequence_length=3),
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    assert collector.last_trace is not None
    assert collector.last_trace.boundaries == ()
    assert collector.last_trace.final_state is not None
    assert collector.last_trace.final_state.position.sequence_length == 3
