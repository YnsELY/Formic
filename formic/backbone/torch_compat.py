"""Environment compatibility shim: torch 2.4 x transformers 5.8.

This is **not** a model change. Transformers 5.8 registers a MoE fallback custom
op whose signature uses postponed (string) annotations; torch 2.4's
``infer_schema`` cannot resolve them and raises at import time. The audit hit the
exact same wall and documented the same fix:

    "PyTorch 2.4 ne sait pas inferer une annotation differee d'un custom op MoE
     de Transformers 5.8; les scripts dynamiques resolvent cette annotation au
     runtime. Ce correctif ne change ni le graphe Qwen, ni les poids, ni les
     calculs."
    -- audits/qwen3_8_27b/README.md

What the shim does: resolve the annotations with ``typing.get_type_hints`` before
handing the function to ``torch.library.custom_op``. Qwen3.5 has no MoE layer, so
the op is never called on our path; only its *registration* has to succeed.

The shim is idempotent, applies only on torch 2.4.x, and its activation is
reported by :func:`formic.science.determinism.environment_report`, so no run can
quote numbers without disclosing that it was active.
"""

from __future__ import annotations

import typing

__all__ = ["ensure_torch_compat", "compat_status"]

_STATUS: dict[str, object] = {
    "torch_version": None,
    "annotation_shim_applied": False,
    "annotation_shim_needed": False,
}


def ensure_torch_compat() -> dict[str, object]:
    """Apply the annotation shim if the installed torch needs it. Idempotent."""
    import torch

    _STATUS["torch_version"] = torch.__version__
    needed = torch.__version__.startswith("2.4")
    _STATUS["annotation_shim_needed"] = needed
    if not needed:
        return dict(_STATUS)

    if getattr(torch.library.custom_op, "_formic_annotation_shim", False):
        _STATUS["annotation_shim_applied"] = True
        return dict(_STATUS)

    original = torch.library.custom_op

    def custom_op_with_resolved_annotations(name, fn=None, /, **kwargs):
        if fn is not None:
            try:
                fn.__annotations__ = typing.get_type_hints(fn)
            except Exception:  # pragma: no cover - leave the original error path
                pass
        return original(name, fn, **kwargs)

    custom_op_with_resolved_annotations._formic_annotation_shim = True  # type: ignore[attr-defined]
    torch.library.custom_op = custom_op_with_resolved_annotations  # type: ignore[assignment]
    _STATUS["annotation_shim_applied"] = True
    return dict(_STATUS)


def compat_status() -> dict[str, object]:
    """Current shim status, for the environment record of a run."""
    return dict(_STATUS)
