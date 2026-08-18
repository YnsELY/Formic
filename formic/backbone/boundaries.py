"""Inert insertion points at the 17 hybrid-group boundaries.

Design rule (plan rule 3 + step-1 checklist): boundaries are *config-driven and
inert by default*. When nothing is enabled, **no PyTorch hook is registered at
all** — not a no-op hook, none — so the forward graph is byte-for-byte the stock
Hugging Face graph and the identity property is structural rather than
empirical.

Two kinds of attachment exist:

``observer``
    Read-only. Sees the residual-stream tensor crossing the boundary and returns
    nothing. This is what step 2 uses to capture per-boundary hidden states.

``insertion``
    May return a replacement tensor. Reserved for step-8 sidecars. With no
    callback registered it is an exact identity (returns ``None``, so Hugging
    Face keeps its own tensor untouched).

Nothing in this module reads or writes the KV/GDN cache: boundary hooks observe
the residual stream only. Cache manipulation is step 2's snapshot/restore
primitive, and it obeys A3/A4 there.

Two runtime facts this relies on, both verified in transformers 5.8:

* ``Qwen3_5TextModel.forward`` calls each layer as
  ``decoder_layer(hidden_states, position_embeddings=..., ...)`` - the residual
  tensor is the first *positional* argument - and the layer returns a single
  tensor, not a tuple.
* ``Qwen3_5DecoderLayer`` derives from ``GradientCheckpointingLayer``, whose
  ``__call__`` delegates to ``nn.Module.__call__``; standard PyTorch hooks
  therefore fire normally, with or without gradient checkpointing.

Both shapes are handled defensively anyway (kwarg or positional, tensor or
tuple) so a future upstream change degrades into an explicit error rather than a
silent no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import torch

from formic.backbone.groups import Boundary, HybridGroupView, get_decoder_layers

__all__ = [
    "BoundaryHookManager",
    "BoundaryEvent",
    "ObserverFn",
    "InsertionFn",
    "BoundaryHookError",
]


class BoundaryHookError(RuntimeError):
    """Raised when hook attachment violates an architecture rule."""


@dataclass(frozen=True)
class BoundaryEvent:
    """Context handed to a boundary callback."""

    boundary: Boundary
    hidden_states: torch.Tensor
    #: ``"pre"`` for boundaries in front of a layer, ``"post"`` for POST_G16.
    kind: str


class ObserverFn(Protocol):
    def __call__(self, event: BoundaryEvent) -> None: ...


class InsertionFn(Protocol):
    def __call__(self, event: BoundaryEvent) -> torch.Tensor | None: ...


@dataclass
class _Attachment:
    name: str
    role: str  # "observer" | "insertion"
    handle: Any = None
    callback: Callable[[BoundaryEvent], Any] | None = None


@dataclass
class BoundaryHookManager:
    """Attach/detach inert boundary hooks on an already-loaded model.

    Usage::

        manager = BoundaryHookManager(model, view)
        manager.configure(observers=("G4_G5",))
        with manager:                     # hooks live only inside the block
            out = model(**inputs)

    With an empty configuration, :meth:`attach` registers nothing and
    :attr:`num_active_hooks` is 0.
    """

    model: Any
    view: HybridGroupView
    observers: tuple[str, ...] = ()
    insertions: tuple[str, ...] = ()
    _attachments: list[_Attachment] = field(default_factory=list, init=False, repr=False)
    _observer_callbacks: dict[str, ObserverFn] = field(default_factory=dict, init=False, repr=False)
    _insertion_callbacks: dict[str, InsertionFn] = field(default_factory=dict, init=False, repr=False)

    # -- configuration -----------------------------------------------------

    def configure(
        self,
        observers: Sequence[str] = (),
        insertions: Sequence[str] = (),
    ) -> "BoundaryHookManager":
        """Select which boundaries become active. Raises on unknown names."""
        if self.is_attached:
            raise BoundaryHookError("configure() called while hooks are attached; detach() first")
        if len(set(observers)) != len(observers):
            raise BoundaryHookError("observer boundary names must be unique")
        if len(set(insertions)) != len(insertions):
            raise BoundaryHookError("insertion boundary names must be unique")
        for name in list(observers) + list(insertions):
            self.view.boundary(name)  # raises ValueError on unknown name
        self.observers = tuple(observers)
        self.insertions = tuple(insertions)
        return self

    @classmethod
    def from_config(cls, model: Any, view: HybridGroupView, config: Any) -> "BoundaryHookManager":
        """Build from a :class:`~formic.config.schema.BoundariesSection`."""
        manager = cls(model=model, view=view)
        return manager.configure(
            observers=getattr(config, "enabled_observers", ()),
            insertions=getattr(config, "enabled_insertions", ()),
        )

    def set_observer(self, name: str, fn: ObserverFn) -> None:
        self.view.boundary(name)
        if name not in self.observers:
            raise BoundaryHookError(
                f"observer callback for {name} is not enabled by the run config"
            )
        self._observer_callbacks[name] = fn

    def set_insertion(self, name: str, fn: InsertionFn) -> None:
        self.view.boundary(name)
        if name not in self.insertions:
            raise BoundaryHookError(
                f"insertion callback for {name} is not enabled by the run config"
            )
        self._insertion_callbacks[name] = fn

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_attached(self) -> bool:
        return bool(self._attachments)

    @property
    def num_active_hooks(self) -> int:
        return len(self._attachments)

    @property
    def active_boundaries(self) -> tuple[str, ...]:
        return tuple(a.name for a in self._attachments)

    def attach(self) -> int:
        """Register hooks for the configured boundaries. Returns the hook count."""
        if self.is_attached:
            raise BoundaryHookError("hooks already attached")
        layers = get_decoder_layers(self.model)
        for name in self.observers:
            self._attach_one(layers, name, role="observer")
        for name in self.insertions:
            self._attach_one(layers, name, role="insertion")
        return self.num_active_hooks

    def detach(self) -> None:
        for attachment in self._attachments:
            if attachment.handle is not None:
                attachment.handle.remove()
        self._attachments.clear()

    def __enter__(self) -> "BoundaryHookManager":
        self.attach()
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # -- internals ---------------------------------------------------------

    def _attach_one(self, layers: Sequence[Any], name: str, role: str) -> None:
        boundary = self.view.boundary(name)
        attachment = _Attachment(name=name, role=role)
        if boundary.before_layer is not None:
            module = layers[boundary.before_layer]
            attachment.handle = module.register_forward_pre_hook(
                self._make_pre_hook(boundary, role), with_kwargs=True
            )
        else:
            module = layers[boundary.after_layer]  # type: ignore[index]
            attachment.handle = module.register_forward_hook(self._make_post_hook(boundary, role))
        self._attachments.append(attachment)

    def _make_pre_hook(self, boundary: Boundary, role: str) -> Callable[..., Any]:
        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            hidden, source = _extract_hidden(args, kwargs)
            event = BoundaryEvent(boundary=boundary, hidden_states=hidden, kind="pre")
            if role == "observer":
                callback = self._observer_callbacks.get(boundary.name)
                if callback is not None:
                    callback(event)
                return None
            replacement = self._apply_insertion(event)
            if replacement is None:
                return None
            return _reinject_hidden(args, kwargs, replacement, source)

        return hook

    def _make_post_hook(self, boundary: Boundary, role: str) -> Callable[..., Any]:
        def hook(module: Any, args: tuple, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            event = BoundaryEvent(boundary=boundary, hidden_states=hidden, kind="post")
            if role == "observer":
                callback = self._observer_callbacks.get(boundary.name)
                if callback is not None:
                    callback(event)
                return None
            replacement = self._apply_insertion(event)
            if replacement is None:
                return None
            if isinstance(output, tuple):
                return (replacement,) + tuple(output[1:])
            return replacement

        return hook

    def _apply_insertion(self, event: BoundaryEvent) -> torch.Tensor | None:
        callback = self._insertion_callbacks.get(event.boundary.name)
        if callback is None:
            # Inert by construction: no callback means exact identity.
            return None
        replacement = callback(event)
        if replacement is None:
            return None
        if replacement.shape != event.hidden_states.shape:
            raise BoundaryHookError(
                f"insertion at {event.boundary.name} changed the hidden shape "
                f"{tuple(event.hidden_states.shape)} -> {tuple(replacement.shape)}"
            )
        if replacement.dtype != event.hidden_states.dtype:
            raise BoundaryHookError(
                f"insertion at {event.boundary.name} changed dtype "
                f"{event.hidden_states.dtype} -> {replacement.dtype}"
            )
        return replacement


def _extract_hidden(args: tuple, kwargs: dict) -> tuple[torch.Tensor, str]:
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"], "kwargs"
    if args:
        return args[0], "args"
    raise BoundaryHookError("decoder layer called without hidden_states")


def _reinject_hidden(
    args: tuple, kwargs: dict, replacement: torch.Tensor, source: str
) -> tuple[tuple, dict]:
    if source == "kwargs":
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = replacement
        return args, new_kwargs
    return (replacement,) + tuple(args[1:]), kwargs


def count_registered_hooks(model: Any) -> int:
    """Total forward/pre-forward hooks registered on the decoder layers.

    Used by the identity guard: with every Formic flag OFF this must be 0.
    """
    total = 0
    for layer in get_decoder_layers(model):
        total += len(getattr(layer, "_forward_pre_hooks", {}))
        total += len(getattr(layer, "_forward_hooks", {}))
    return total
