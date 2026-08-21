"""Deterministic SPEC-02 prefill segmentations."""

from __future__ import annotations

import math

SEGMENTATIONS = ("early", "median", "late", "quarters")


def segment_slices(length: int, strategy: str) -> tuple[slice, ...]:
    if length < 2:
        raise ValueError("segmented prefill requires at least two tokens")
    if strategy == "early":
        cuts = (1,)
    elif strategy == "median":
        cuts = (length // 2,)
    elif strategy == "late":
        cuts = (length - 1,)
    elif strategy == "quarters":
        width = math.ceil(length / 4)
        cuts = tuple(range(width, length, width))
    else:
        raise ValueError(f"unknown segmentation {strategy!r}")
    points = (0, *cuts, length)
    return tuple(slice(start, stop) for start, stop in zip(points, points[1:]))
