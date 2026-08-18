"""Boundary hooks must be inert by default (step-1 checklist).

Weight-free: the mechanism is exercised on a stub stack that mimics the shape of
the real decoder (a ``model.layers`` ModuleList of 64 layers taking and
returning ``hidden_states``). Verification against the *real* modules happens in
the acceptance script and, formally, in step 2.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from formic.backbone import constants as C
from formic.backbone.boundaries import (
    BoundaryHookError,
    BoundaryHookManager,
    count_registered_hooks,
)
from formic.backbone.groups import HybridGroupView
from formic.config.schema import BoundariesSection


class _StubLayer(nn.Module):
    """Adds a constant so any tampering with the residual stream is visible."""

    def __init__(self, index: int, layer_type: str) -> None:
        super().__init__()
        self.index = index
        self.layer_type = layer_type

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states + 1.0


class _StubInner(nn.Module):
    def __init__(self, types) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_StubLayer(i, t) for i, t in enumerate(types))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states)
        return hidden_states


class _StubModel(nn.Module):
    def __init__(self, types) -> None:
        super().__init__()
        self.model = _StubInner(types)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


@pytest.fixture()
def view() -> HybridGroupView:
    return HybridGroupView(C.expected_layer_types())


@pytest.fixture()
def model(view: HybridGroupView) -> _StubModel:
    return _StubModel(view.layer_types)


def test_default_config_registers_no_hook_at_all(model, view):
    manager = BoundaryHookManager.from_config(model, view, BoundariesSection())
    assert manager.attach() == 0
    assert manager.num_active_hooks == 0
    assert count_registered_hooks(model) == 0


def test_baseline_forward_is_untouched_without_hooks(model, view):
    hidden = torch.zeros(1, 2, 8)
    assert torch.equal(model(hidden), hidden + C.NUM_LAYERS)


def test_observer_sees_the_residual_stream_at_the_right_boundaries(model, view):
    seen: dict[str, float] = {}

    manager = BoundaryHookManager(model, view).configure(
        observers=("PRE_G1", "G1_G2", "G4_G5", "POST_G16")
    )
    for name in ("PRE_G1", "G1_G2", "G4_G5", "POST_G16"):
        manager.set_observer(name, lambda event: seen.__setitem__(
            event.boundary.name, float(event.hidden_states.flatten()[0])
        ))

    hidden = torch.zeros(1, 1, 4)
    with manager:
        assert manager.num_active_hooks == 4
        assert count_registered_hooks(model) == 4
        out = model(hidden)

    # Each stub layer adds 1: the value at a boundary equals the layers crossed.
    assert seen == {"PRE_G1": 0.0, "G1_G2": 4.0, "G4_G5": 16.0, "POST_G16": 64.0}
    assert float(out.flatten()[0]) == 64.0
    assert count_registered_hooks(model) == 0  # detached on context exit


def test_observers_cannot_change_the_output(model, view):
    hidden = torch.zeros(1, 1, 4)
    reference = model(hidden).clone()

    manager = BoundaryHookManager(model, view).configure(observers=tuple(
        b.name for b in view.boundaries
    ))
    for boundary in view.boundaries:
        manager.set_observer(boundary.name, lambda event: None)
    with manager:
        assert manager.num_active_hooks == C.NUM_BOUNDARIES == 17
        observed = model(hidden)
    assert torch.equal(observed, reference)


def test_insertion_without_callback_is_exact_identity(model, view):
    hidden = torch.zeros(1, 1, 4)
    reference = model(hidden).clone()
    manager = BoundaryHookManager(model, view).configure(
        insertions=tuple(b.name for b in view.boundaries)
    )
    with manager:
        assert manager.num_active_hooks == 17
        result = model(hidden)
    assert torch.equal(result, reference)


def test_insertion_with_callback_changes_the_stream(model, view):
    hidden = torch.zeros(1, 1, 4)
    manager = BoundaryHookManager(model, view).configure(insertions=("G8_G9",))
    manager.set_insertion("G8_G9", lambda event: event.hidden_states + 100.0)
    with manager:
        result = model(hidden)
    assert float(result.flatten()[0]) == 64.0 + 100.0


def test_callback_cannot_activate_a_boundary_not_selected_by_config(model, view):
    manager = BoundaryHookManager(model, view)
    with pytest.raises(BoundaryHookError):
        manager.set_observer("PRE_G1", lambda event: None)
    with pytest.raises(BoundaryHookError):
        manager.set_insertion("POST_G16", lambda event: event.hidden_states)


def test_insertion_at_the_final_boundary_is_applied(model, view):
    hidden = torch.zeros(1, 1, 4)
    manager = BoundaryHookManager(model, view).configure(insertions=("POST_G16",))
    manager.set_insertion("POST_G16", lambda event: event.hidden_states * 2)
    with manager:
        result = model(hidden)
    assert float(result.flatten()[0]) == 128.0


def test_insertion_cannot_change_shape_or_dtype(model, view):
    manager = BoundaryHookManager(model, view).configure(insertions=("G2_G3",))
    manager.set_insertion("G2_G3", lambda event: event.hidden_states[:, :, :2])
    with pytest.raises(BoundaryHookError):
        with manager:
            model(torch.zeros(1, 1, 4))
    manager.detach()


def test_unknown_boundary_is_rejected(model, view):
    with pytest.raises(ValueError):
        BoundaryHookManager(model, view).configure(observers=("NOT_A_BOUNDARY",))


def test_duplicate_boundaries_are_rejected(model, view):
    with pytest.raises(BoundaryHookError, match="unique"):
        BoundaryHookManager(model, view).configure(
            insertions=("PRE_G1", "PRE_G1")
        )


def test_configure_while_attached_is_rejected(model, view):
    manager = BoundaryHookManager(model, view).configure(observers=("PRE_G1",))
    manager.attach()
    try:
        with pytest.raises(BoundaryHookError):
            manager.configure(observers=("G1_G2",))
    finally:
        manager.detach()


def test_detach_removes_every_hook(model, view):
    manager = BoundaryHookManager(model, view).configure(
        observers=("PRE_G1", "G8_G9"), insertions=("POST_G16",)
    )
    manager.attach()
    assert count_registered_hooks(model) == 3
    manager.detach()
    assert count_registered_hooks(model) == 0
    assert manager.num_active_hooks == 0
