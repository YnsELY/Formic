"""Environment pinning.

Plan 2.4: bit-exactness is only required at identical config *and backend*, so
every run records the backend it actually ran on. This module produces that
record; nothing here changes numerics by itself.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "configure_determinism",
    "environment_report",
    "git_commit",
    "git_dirty",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_determinism(seed: int, deterministic: bool = True) -> None:
    """Apply the run's RNG and backend policy before any model execution."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    # Benchmarking is not a configured behavior and can select a different
    # convolution algorithm based on timing noise, so it remains disabled.
    torch.backends.cudnn.benchmark = False


def git_commit(repo: str | Path | None = None) -> str | None:
    return _git(("rev-parse", "HEAD"), repo)


def git_dirty(repo: str | Path | None = None) -> bool | None:
    status = _git(("status", "--porcelain"), repo)
    if status is None:
        return None
    return bool(status.strip())


def _git(args: tuple[str, ...], repo: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=str(repo or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def environment_report() -> dict[str, Any]:
    """Everything needed to decide whether two runs are numerically comparable."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "env": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM")
            if key in os.environ
        },
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["gpu_count"] = torch.cuda.device_count()
            report["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
        report["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        report["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except ImportError:  # pragma: no cover
        report["torch"] = None

    from formic.backbone.torch_compat import compat_status

    report["torch_compat"] = compat_status()

    for module_name in ("transformers", "accelerate", "safetensors", "numpy"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", None)
        except ImportError:  # pragma: no cover
            report[module_name] = None

    # Fast paths deliberately absent from the audit environment: their presence
    # changes GDN/attention kernels and therefore the numerics baseline.
    for optional in ("flash_linear_attention", "fla", "causal_conv1d", "flash_attn"):
        try:
            __import__(optional)
            report[f"has_{optional}"] = True
        except Exception:
            report[f"has_{optional}"] = False
    return report
