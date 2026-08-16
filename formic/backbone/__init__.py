"""Backbone integration: Qwen3.8-27B as an untouched neural substrate.

Public surface:

``constants``
    Audited facts of the checkpoint. The single source of truth for every
    structural number in Formic.
``inventory``
    Strict tensor inventory (A12): permissive loading is impossible, exclusions
    are declared and counted.
``loader``
    Loads the *stock* Hugging Face implementation. Text-only mode uses
    ``Qwen3_5ForCausalLM`` so the vision tower is never constructed (A7).
``groups``
    The 16 hybrid groups as a **view** over intact modules (A11).
``boundaries``
    The 17 inert insertion points; nothing is registered while flags are OFF.
``runner``
    Native generation with pinned thinking/sampling policies.
``torch_compat``
    Environment shim (torch 2.4 x transformers 5.8); touches no Qwen code.
"""

from __future__ import annotations

__all__ = [
    "constants",
    "inventory",
    "loader",
    "groups",
    "boundaries",
    "runner",
    "torch_compat",
]
