"""Explicit post-pod promotion of reviewed SPEC-02 calibration evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from formic.science.identity.artifacts import atomic_write_json
from formic.science.identity.calibration import CalibrationError, load_candidate
from formic.science.identity.tolerances import load_tolerances


class PromotionError(RuntimeError):
    pass


def promote_candidate_tolerances(
    *,
    candidate_path: str | Path,
    raw_measurements_path: str | Path,
    justifications_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a governed-format tolerance table after human review.

    The explicit justifications file prevents a pod run from assigning causal
    explanations.  This function writes no ADR, report, verdict, or Git commit.
    Those remain deliberate human actions.
    """
    candidate = load_candidate(candidate_path)
    raw = Path(raw_measurements_path)
    if not raw.is_file():
        raise PromotionError("raw measurement artefact is missing")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    if raw_sha != candidate["raw_measurements_sha256"]:
        raise PromotionError("raw measurement hash differs from candidate")
    try:
        justifications = json.loads(Path(justifications_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError("invalid human justifications JSON") from exc
    if not isinstance(justifications, dict):
        raise PromotionError("human justifications must be an object")

    records: list[dict[str, Any]] = []
    for candidate_record in candidate["records"]:
        key = _record_key(candidate_record)
        criterion = candidate_record["criterion"]
        if criterion not in ("exact", "bounded"):
            raise PromotionError(f"unknown candidate criterion for {key}")
        if criterion == "bounded":
            justification = justifications.get(key)
            if not isinstance(justification, str) or not justification.strip():
                raise PromotionError(f"bounded candidate {key} lacks human justification")
        else:
            justification = None
        observations = candidate_record["observations"]
        exact_lengths = sorted({int(item["exact_prompt_length"]) for item in observations})
        records.append(
            {
                "mode": candidate_record["mode"],
                "point": candidate_record["point"],
                "length_class": candidate_record["length_class"],
                "criterion": criterion,
                "max_abs_delta": candidate_record["max_abs_delta"],
                "physical_justification": justification,
                "evidence": {
                    "exact_lengths": exact_lengths,
                    "observations": [
                        {
                            "continuation_seed": item["continuation_seed"],
                            "repetition": item["repetition"],
                            "max_abs_delta": item["max_abs_delta"],
                            # The economical reference-floor phase records
                            # logits only.  Non-logit exact rows have zero
                            # measured delta; bounded rows require human
                            # review before reaching this promotion step.
                            "reference_floor": 0.0,
                        }
                        for item in observations
                    ],
                    "measurement_artifact": str(raw),
                    "measurement_artifact_sha256": raw_sha,
                },
            }
        )
    value = {
        "schema_version": 1,
        "status": "calibrated",
        "margin_multiplier": 2.0,
        "records": records,
    }
    atomic_write_json(output_path, value)
    try:
        load_tolerances(output_path)
    except Exception as exc:
        raise PromotionError(f"promoted tolerance table failed strict validation: {exc}") from exc
    return value


def _record_key(value: dict[str, Any]) -> str:
    return f"{value['mode']}/{value['point']}/{value['length_class']}"
