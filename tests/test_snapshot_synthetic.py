"""SPEC-02 snapshot/restore tests on synthetic audited hybrid caches."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from formic.backbone import constants as C
from formic.state.snapshot import (
    BranchActivationError,
    ExecutionStateController,
    PositionState,
    iter_snapshot_tensors,
    snapshot,
    snapshot_fingerprint,
    tensor_storage_identity,
)
from tests.toy_qwen import (
    CacheModelStub,
    audited_text_config,
    cache_tensors,
    synthetic_cache,
    toy_text_config,
    toy_model,
)


def _position(length: int) -> PositionState:
    return PositionState(
        sequence_length=length,
        cache_position=torch.arange(length, dtype=torch.long),
        position_ids=torch.arange(length, dtype=torch.long).view(1, -1),
        attention_mask=torch.ones(1, length, dtype=torch.long),
    )


def _storage_set(tensors) -> set[tuple[str, int, int]]:
    return {tensor_storage_identity(tensor) for tensor in tensors}


def test_audited_synthetic_cache_has_real_shapes_and_hybrid_layout():
    config = audited_text_config()
    cache = synthetic_cache(config, sequence_length=2)
    assert len(cache.layers) == C.NUM_LAYERS
    assert cache.get_seq_length() == 2

    for index, layer in enumerate(cache.layers):
        if index in C.ATTENTION_LAYER_INDICES:
            assert tuple(layer.keys.shape) == (1, C.NUM_KEY_VALUE_HEADS, 2, C.HEAD_DIM)
            assert tuple(layer.values.shape) == tuple(layer.keys.shape)
        else:
            assert tuple(layer.conv_states.shape) == (1, *C.GDN_CONV_STATE_SHAPE)
            assert tuple(layer.recurrent_states.shape) == (
                1,
                *C.GDN_RECURRENT_STATE_SHAPE,
            )
            assert layer.has_previous_state is True
        for tensor in cache_tensors(type("One", (), {"layers": [layer]})()).values():
            assert tensor.dtype == torch.bfloat16


def test_capture_clones_source_and_records_absent_text_model_state():
    config = toy_text_config()
    model = CacheModelStub(config)
    cache = synthetic_cache(config)
    state = snapshot(model=model, cache=cache, position=_position(3))

    assert len(state.layers) == 64
    assert state.position.sequence_length == 3
    assert state.model_state[0].attribute == "rope_deltas"
    assert state.model_state[0].status == "absent"

    source_storage = _storage_set(cache_tensors(cache).values())
    snapshot_storage = _storage_set(tensor for _, tensor in iter_snapshot_tensors(state))
    assert source_storage.isdisjoint(snapshot_storage)


def test_two_restores_share_no_storage_and_snapshot_is_immutable():
    config = toy_text_config()
    model = CacheModelStub(config)
    source = synthetic_cache(config)
    frozen = snapshot(model=model, cache=source, position=_position(3))
    frozen_before = snapshot_fingerprint(frozen)

    controller = ExecutionStateController(model)
    branch_a = controller.restore(frozen)
    branch_b = controller.restore(frozen)

    snapshot_storage = _storage_set(tensor for _, tensor in iter_snapshot_tensors(frozen))
    a_tensors = cache_tensors(branch_a.cache)
    b_tensors = cache_tensors(branch_b.cache)
    a_storage = _storage_set(a_tensors.values())
    b_storage = _storage_set(b_tensors.values())
    assert a_storage.isdisjoint(snapshot_storage)
    assert b_storage.isdisjoint(snapshot_storage)
    assert a_storage.isdisjoint(b_storage)

    b_before = {name: tensor.clone() for name, tensor in b_tensors.items()}
    with pytest.raises(BranchActivationError, match="inactive branch"):
        controller.forward(branch_a, input_ids=torch.tensor([[7]]), use_cache=True)

    controller.activate(branch_a)
    controller.forward(branch_a, input_ids=torch.tensor([[7]]), use_cache=True)
    assert any(not torch.equal(a_tensors[name], b_tensors[name]) for name in a_tensors)
    assert all(torch.equal(b_tensors[name], b_before[name]) for name in b_tensors)
    assert snapshot_fingerprint(frozen) == frozen_before

    controller.activate(branch_b)
    controller.forward(branch_b, input_ids=torch.tensor([[8]]), use_cache=True)
    assert not torch.equal(b_tensors["layers[0].recurrent_states"], b_before["layers[0].recurrent_states"])
    assert snapshot_fingerprint(frozen) == frozen_before


def test_model_attached_tensor_state_is_cloned_per_activation():
    config = toy_text_config()
    model = CacheModelStub(config, rope_deltas=torch.tensor([[3, 4]]))
    frozen = snapshot(
        model=model, cache=synthetic_cache(config), position=_position(3)
    )
    controller = ExecutionStateController(model)
    a = controller.restore(frozen)
    a_ptr = model.rope_deltas.untyped_storage().data_ptr()
    b = controller.restore(frozen)
    b_ptr = model.rope_deltas.untyped_storage().data_ptr()
    assert a_ptr != b_ptr
    assert model.rope_deltas.untyped_storage().data_ptr() != (
        frozen.model_state[0].value.untyped_storage().data_ptr()
    )
    controller.activate(a)
    assert torch.equal(model.rope_deltas, torch.tensor([[3, 4]]))
    controller.activate(b)


def test_position_tensors_are_cloned_for_every_consumer():
    config = toy_text_config()
    model = CacheModelStub(config)
    position = _position(3)
    frozen = snapshot(model=model, cache=synthetic_cache(config), position=position)
    controller = ExecutionStateController(model)
    a = controller.restore(frozen)
    b = controller.restore(frozen)
    assert a.position.cache_position.data_ptr() != b.position.cache_position.data_ptr()
    assert a.position.cache_position.data_ptr() != frozen.position.cache_position.data_ptr()
    a.position.cache_position.add_(10)
    assert torch.equal(b.position.cache_position, torch.arange(3))


def test_controller_rejects_foreign_branch_and_batching():
    config = toy_text_config()
    model_a = CacheModelStub(config)
    model_b = CacheModelStub(config)
    frozen = snapshot(
        model=model_a, cache=synthetic_cache(config), position=_position(3)
    )
    controller_a = ExecutionStateController(model_a)
    controller_b = ExecutionStateController(model_b)
    branch = controller_a.restore(frozen)
    with pytest.raises(BranchActivationError, match="different controller"):
        controller_b.activate(branch)
    with pytest.raises(BranchActivationError, match="batch 1"):
        controller_a.forward(branch, input_ids=torch.ones(2, 1, dtype=torch.long))


def test_snapshot_module_never_uses_crop_or_bare_dynamic_cache():
    path = Path(__file__).resolve().parents[1] / "formic" / "state" / "snapshot.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "crop", "A3 forbids crop() for hybrid rollback"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "DynamicCache":
                assert any(keyword.arg == "config" for keyword in node.keywords), (
                    "A2 forbids DynamicCache() without the model config"
                )


def _forced_toy_decode(model, prompt: torch.Tensor, forced: tuple[int, ...]):
    past = None
    current = prompt
    logits: list[torch.Tensor] = []
    for token_id in forced:
        output = model(input_ids=current, past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits.append(output.logits[0, -1].detach().clone())
        current = torch.tensor([[token_id]], dtype=torch.long)
    return logits, past


def test_toy_continuity_matches_snapshot_restore_resume_exactly():
    model = toy_model(seed=17)
    prompt = torch.tensor([[4, 9, 2, 11]], dtype=torch.long)
    forced = (13, 5, 21, 8)

    continuous_logits, continuous_cache = _forced_toy_decode(model, prompt, forced)

    first = model(input_ids=prompt, use_cache=True)
    interrupted_logits = [first.logits[0, -1].detach().clone()]
    second = model(
        input_ids=torch.tensor([[forced[0]]], dtype=torch.long),
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    interrupted_logits.append(second.logits[0, -1].detach().clone())

    frozen = snapshot(
        model=model,
        cache=second.past_key_values,
        position=PositionState(sequence_length=int(second.past_key_values.get_seq_length())),
    )
    controller = ExecutionStateController(model)
    branch = controller.restore(frozen)
    for token_id in forced[2:]:
        output = controller.forward(
            branch,
            input_ids=torch.tensor([[forced[len(interrupted_logits) - 1]]]),
            use_cache=True,
        )
        interrupted_logits.append(output.logits[0, -1].detach().clone())

    assert len(interrupted_logits) == len(continuous_logits)
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(continuous_logits, interrupted_logits)
    )
    expected_cache = cache_tensors(continuous_cache)
    restored_cache = cache_tensors(branch.cache)
    assert expected_cache.keys() == restored_cache.keys()
    assert all(
        torch.equal(expected_cache[name], restored_cache[name])
        for name in expected_cache
    )
