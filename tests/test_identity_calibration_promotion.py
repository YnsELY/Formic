from __future__ import annotations

import json

import pytest

from formic.science.identity.artifacts import atomic_write_json
from formic.science.identity.calibration import CalibrationError, build_candidate_tolerances
from formic.science.identity.promotion import promote_candidate_tolerances
from formic.science.identity.tolerances import load_tolerances
from formic.science.identity.types import ComparisonPoint, ExecutionMode


def _observation(repetition: int, delta: float) -> dict:
    return {
        "case_id": f"case-{repetition}",
        "prompt_id": "short_error_assertion",
        "length_class": "short",
        "exact_prompt_length": 26,
        "mode": "decode_cached",
        "segmentation": None,
        "sampling": "greedy",
        "continuation_seed": None,
        "repetition": repetition,
        "measurements": [
            {
                "step": 0,
                "location": {"point": "logits", "boundary": None, "layer": None, "component": None},
                "metric": {
                    "tensor": {"exact": delta == 0.0, "max_abs_delta": delta},
                    "top1_agreement": True,
                },
            }
        ],
    }


def test_candidate_uses_three_repetitions_and_promotion_requires_human_justification(tmp_path):
    raw_path = tmp_path / "raw_measurements.json"
    observations = [_observation(0, 0.0), _observation(1, 0.125), _observation(2, 0.25)]
    atomic_write_json(raw_path, {"schema_version": 1, "observations": observations})
    import hashlib

    candidate = build_candidate_tolerances(
        observations,
        raw_measurements_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        reference_floor_observations=[
            {
                "point": "logits",
                "repetition": repetition,
                "max_abs_delta": 0.0,
            }
            for repetition in range(3)
        ],
    )
    candidate_path = tmp_path / "candidate.json"
    atomic_write_json(candidate_path, candidate)
    justifications = tmp_path / "justifications.json"
    justifications.write_text(
        json.dumps({"decode_cached/logits/short": "Measured BF16 boundary rounding."}),
        encoding="utf-8",
    )
    output = tmp_path / "tolerances.json"
    promote_candidate_tolerances(
        candidate_path=candidate_path,
        raw_measurements_path=raw_path,
        justifications_path=justifications,
        output_path=output,
    )
    catalogue = load_tolerances(output)
    record = catalogue.threshold(ExecutionMode.DECODE_CACHED, ComparisonPoint.LOGITS, "short")
    assert record.criterion == "bounded"
    assert record.max_abs_delta == 0.5


def test_reference_floor_is_carried_into_candidate_and_promoted_evidence(tmp_path):
    raw_path = tmp_path / "raw_measurements.json"
    observations = [_observation(index, 0.125) for index in range(3)]
    atomic_write_json(raw_path, {"schema_version": 1, "observations": observations})
    import hashlib

    candidate = build_candidate_tolerances(
        observations,
        raw_measurements_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        reference_floor_observations=[
            {
                "point": "logits",
                "repetition": repetition,
                "max_abs_delta": 0.75,
            }
            for repetition in range(3)
        ],
    )
    assert candidate["records"][0]["max_abs_delta"] == 0.75
    assert candidate["records"][0]["reference_floor_max_abs_delta"] == 0.75
    assert all(
        item["reference_floor"] == 0.75
        for item in candidate["records"][0]["observations"]
    )


def test_candidate_rejects_reference_floor_with_fewer_than_three_repetitions():
    with pytest.raises(CalibrationError, match="three measured repetitions"):
        build_candidate_tolerances(
            [_observation(index, 0.0) for index in range(3)],
            raw_measurements_sha256="a" * 64,
            reference_floor_observations=[
                {"point": "logits", "repetition": index, "max_abs_delta": 0.0}
                for index in range(2)
            ],
        )
