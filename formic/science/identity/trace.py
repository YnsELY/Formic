"""Read-only capture at the 17 audited group boundaries.

For short and medium inputs, each natural exit boundary captures only the four
cache layers of the group that has just completed. The cumulative cache is
never copied 17 times. Recompute mode has no cache and records GDN/KV as
``not_applicable``. Long inputs use final-state-only capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from formic.backbone.boundaries import BoundaryEvent, BoundaryHookManager
from formic.backbone.groups import BOUNDARY_NAMES, HybridGroupView
from formic.science.identity.types import CaptureProfile
from formic.state.snapshot import (
    ExecutionSnapshot,
    LayerSnapshot,
    ModelStateSlot,
    PositionState,
    capture_cache_layers,
    capture_model_state,
    snapshot,
)


@dataclass(frozen=True)
class BoundaryCapture:
    name: str
    hidden_states: torch.Tensor
    completed_group: int | None
    cache_applicability: str
    cache_layers: tuple[LayerSnapshot, ...]


@dataclass(frozen=True)
class ForwardTrace:
    logits: torch.Tensor
    boundaries: tuple[BoundaryCapture, ...]
    model_state: tuple[ModelStateSlot, ...]
    final_state: ExecutionSnapshot | None


class IdentityTraceCollector:
    """Explicit opt-in observer; constructing it does not attach a hook."""

    def __init__(
        self,
        *,
        model: Any,
        view: HybridGroupView,
        cache: Any | None,
        capture_profile: CaptureProfile,
    ) -> None:
        self.model = model
        self.view = view
        self.cache = cache
        self.capture_profile = capture_profile
        observers = BOUNDARY_NAMES if capture_profile is CaptureProfile.FULL_BOUNDARIES else ()
        self.manager = BoundaryHookManager(model, view).configure(observers=observers)
        self._captures: list[BoundaryCapture] = []
        self.last_trace: ForwardTrace | None = None
        for name in observers:
            self.manager.set_observer(name, self._observe)

    def __enter__(self) -> "IdentityTraceCollector":
        self._captures.clear()
        self.manager.attach()
        return self

    def __exit__(self, *exc: object) -> None:
        self.manager.detach()

    def _observe(self, event: BoundaryEvent) -> None:
        group_index = event.boundary.upstream_group
        layers: tuple[LayerSnapshot, ...] = ()
        applicability = "not_applicable"
        if group_index is not None and self.cache is not None:
            layers = capture_cache_layers(
                self.cache, self.view.group(group_index).layer_indices
            )
            applicability = "applicable"
        self._captures.append(
            BoundaryCapture(
                name=event.boundary.name,
                hidden_states=event.hidden_states.detach().clone(
                    memory_format=torch.preserve_format
                ),
                completed_group=group_index,
                cache_applicability=applicability,
                cache_layers=layers,
            )
        )

    def finish(self, outputs: Any, position: PositionState | None) -> ForwardTrace:
        if self.manager.is_attached:
            raise RuntimeError("finish() must be called after trace hooks are detached")
        final_state = None
        if self.capture_profile is CaptureProfile.FINAL_STATE_ONLY:
            if self.cache is not None:
                if position is None:
                    raise ValueError("final-state capture requires explicit position metadata")
                final_state = snapshot(model=self.model, cache=self.cache, position=position)
        if self.capture_profile is CaptureProfile.FULL_BOUNDARIES:
            names = tuple(item.name for item in self._captures)
            if names != BOUNDARY_NAMES:
                raise RuntimeError(f"expected 17 ordered boundaries, got {names}")
        logits = outputs.logits[0, -1].detach().clone(memory_format=torch.preserve_format)
        return ForwardTrace(
            logits=logits,
            boundaries=tuple(self._captures),
            model_state=(
                ()
                if self.capture_profile is CaptureProfile.LOGITS_ONLY
                else capture_model_state(self.model)
            ),
            final_state=final_state,
        )
