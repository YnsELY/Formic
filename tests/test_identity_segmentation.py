from __future__ import annotations

import pytest

from formic.science.identity.segmentation import segment_slices


@pytest.mark.parametrize(
    ("length", "strategy", "expected"),
    [
        (9, "early", ((0, 1), (1, 9))),
        (9, "median", ((0, 4), (4, 9))),
        (9, "late", ((0, 8), (8, 9))),
        (9, "quarters", ((0, 3), (3, 6), (6, 9))),
        (10, "quarters", ((0, 3), (3, 6), (6, 9), (9, 10))),
    ],
)
def test_deterministic_segmentations(length, strategy, expected):
    slices = segment_slices(length, strategy)
    assert tuple((item.start, item.stop) for item in slices) == expected
    assert [index for item in slices for index in range(item.start, item.stop)] == list(
        range(length)
    )


def test_segmentation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        segment_slices(1, "median")
    with pytest.raises(ValueError):
        segment_slices(8, "unknown")
