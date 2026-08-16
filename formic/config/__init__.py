"""Run configuration: strict schema and YAML loader.

A run is fully described by its config (plan rule: reproducibility). Unknown keys
are fatal, every Formic mechanism is a flag defaulting to OFF, and part-2 flags
are refused during part 1.
"""

from __future__ import annotations

__all__ = ["schema", "loader"]
