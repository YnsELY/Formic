from __future__ import annotations

import json

import pytest

from formic.science.identity.tolerances import ToleranceError, load_tolerances
from formic.science.identity.types import ComparisonPoint, ExecutionMode


def _observations(delta: float, floor: float = 0.0):
    return [
        {
            "continuation_seed": 0,
            "repetition": repetition,
            "max_abs_delta": delta,
            "reference_floor": floor,
        }
        for repetition in (0, 1, 2)
    ]


def _catalogue(*, criterion="bounded", threshold=0.5, delta=0.25, floor=0.1):
    return {
        "schema_version": 1,
        "status": "calibrated",
        "margin_multiplier": 2.0,
        "records": [
            {
                "mode": "decode_cached",
                "point": "logits",
                "length_class": "short",
                "criterion": criterion,
                "max_abs_delta": threshold,
                "physical_justification": "measured BF16 recurrent rounding"
                if criterion == "bounded"
                else None,
                "evidence": {
                    "exact_lengths": [17, 23],
                    "observations": _observations(delta, floor),
                    "measurement_artifact": "artifacts/step2/calibration.json",
                    "measurement_artifact_sha256": "a" * 64,
                },
            }
        ],
    }


def _write(tmp_path, value):
    path = tmp_path / "tolerances.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_bounded_tolerance_is_loaded_with_evidence_and_hash(tmp_path):
    catalogue = load_tolerances(_write(tmp_path, _catalogue()))
    record = catalogue.threshold(
        ExecutionMode.DECODE_CACHED, ComparisonPoint.LOGITS, "short"
    )
    assert record.max_abs_delta == 0.5
    assert record.evidence.observed_max == 0.25
    assert len(catalogue.source_sha256) == 64


def test_exact_is_default_shape_but_requires_zero_observations(tmp_path):
    catalogue = load_tolerances(
        _write(tmp_path, _catalogue(criterion="exact", threshold=0, delta=0, floor=0))
    )
    assert catalogue.records[0].criterion == "exact"
    with pytest.raises(ToleranceError, match="all observations"):
        load_tolerances(
            _write(tmp_path, _catalogue(criterion="exact", threshold=0, delta=0.1, floor=0))
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "unknown"),
        (
            lambda value: value["records"][0].update(max_abs_delta=0.49),
            "required",
        ),
        (
            lambda value: value["records"][0]["evidence"].update(
                observations=_observations(0.25)[:2]
            ),
            "at least 3 repetitions",
        ),
        (
            lambda value: value["records"][0].update(physical_justification=None),
            "physical justification",
        ),
    ],
)
def test_invalid_or_under_evidenced_catalogues_are_rejected(tmp_path, mutation, message):
    value = _catalogue()
    mutation(value)
    with pytest.raises(ToleranceError, match=message):
        load_tolerances(_write(tmp_path, value))
