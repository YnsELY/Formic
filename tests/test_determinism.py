"""The YAML deterministic policy must be applied before model execution."""

from __future__ import annotations

import torch

from formic.config.schema import NumericsSection
from formic.science.determinism import configure_determinism, prepare_backend_environment


def test_configure_determinism_seeds_torch_and_pins_cudnn():
    configure_determinism(7, deterministic=True)
    first = torch.rand(4)
    configure_determinism(7, deterministic=True)
    second = torch.rand(4)

    assert torch.equal(first, second)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    configure_determinism(7, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False


def test_configure_determinism_applies_pinned_numerics(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_determinism(7, numerics=NumericsSection())
    assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_prepare_backend_environment_refuses_conflicting_cublas_policy(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with __import__("pytest").raises(RuntimeError, match="conflicts"):
        prepare_backend_environment(NumericsSection())
