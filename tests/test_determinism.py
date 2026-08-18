"""The YAML deterministic policy must be applied before model execution."""

from __future__ import annotations

import torch

from formic.science.determinism import configure_determinism


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
