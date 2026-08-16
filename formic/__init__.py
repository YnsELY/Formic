"""Formic - a software-engineering executor built on the Qwen3.8-27B substrate.

Part 1 scope: integrate the checkpoint without changing its behaviour, expose the
16 hybrid groups as a view, and put the project's scientific tooling in place.

Importing ``formic`` applies one environment shim (torch 2.4 x transformers 5.8
custom-op annotations) *before* any transformers import can fail on it. The shim
touches neither the Qwen graph, nor its weights, nor its computations, and its
activation is recorded in every run's environment report. See
``formic/backbone/torch_compat.py`` and ``docs/adr/ADR-0003-torch-compat-shim.md``.
"""

from __future__ import annotations

from formic.backbone.torch_compat import ensure_torch_compat as _ensure_torch_compat

__version__ = "0.1.0"

_TORCH_COMPAT = _ensure_torch_compat()

__all__ = ["__version__"]
