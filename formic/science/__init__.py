"""Scientific tooling: experiment registry and environment pinning.

Every quoted number must be traceable to (config hash, git commit, seeds,
environment, ``EXP-...`` entry).
"""

from __future__ import annotations

__all__ = ["registry", "determinism"]
