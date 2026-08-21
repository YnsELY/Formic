"""Formal SPEC-02 identity gate primitives."""

from formic.science.identity.metrics import LogitComparison, TensorComparison
from formic.science.identity.types import ComparisonLocation, ExecutionMode

__all__ = [
    "ComparisonLocation",
    "ExecutionMode",
    "LogitComparison",
    "TensorComparison",
]
