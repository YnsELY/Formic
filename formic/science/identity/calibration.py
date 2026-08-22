"""Materialise review-required tolerance candidates from raw A40 observations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from formic.science.identity.artifacts import canonical_json_bytes, sha256_bytes


class CalibrationError(RuntimeError):
    pass


def build_candidate_tolerances(
    observations: Iterable[dict[str, Any]],
    *,
    raw_measurements_sha256: str,
) -> dict[str, Any]:
    """Build a non-governing tolerance candidate from measured observations.

    Exact rows need no human wording.  Bounded rows remain explicitly
    ``REVIEW_REQUIRED``: the campaign records numbers but never assigns a
    causal explanation in the pod session.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        length_class = observation["length_class"]
        if length_class not in ("short", "medium", "long"):
            continue
        for measurement in observation["measurements"]:
            key = (observation["mode"], measurement["location"]["point"], length_class)
            grouped[key].append(
                {
                    "prompt_id": observation["prompt_id"],
                    "exact_prompt_length": observation["exact_prompt_length"],
                    "segmentation": observation["segmentation"],
                    "sampling": observation["sampling"],
                    "continuation_seed": observation["continuation_seed"],
                    "repetition": observation["repetition"],
                    "max_abs_delta": _metric_delta(measurement["metric"]),
                    "top1_agreement": _top1(measurement["metric"]),
                }
            )
    if not grouped:
        raise CalibrationError("no calibration observations were supplied")

    records: list[dict[str, Any]] = []
    for key in sorted(grouped):
        mode, point, length_class = key
        items = grouped[key]
        repetitions = {int(item["repetition"]) for item in items}
        if len(repetitions) < 3:
            raise CalibrationError(f"{key} lacks three measured repetitions")
        observed_max = max(float(item["max_abs_delta"]) for item in items)
        exact = observed_max == 0.0 and all(item["top1_agreement"] is not False for item in items)
        records.append(
            {
                "mode": mode,
                "point": point,
                "length_class": length_class,
                "criterion": "exact" if exact else "bounded",
                "max_abs_delta": 0.0 if exact else 2.0 * observed_max,
                "physical_justification": None if exact else "REVIEW_REQUIRED",
                "observations": items,
                "observed_max_abs_delta": observed_max,
            }
        )
    return {
        "schema_version": 1,
        "status": "candidate_review_required",
        "margin_multiplier": 2.0,
        "raw_measurements_sha256": raw_measurements_sha256,
        "records": records,
    }


def candidate_verdict(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report only hard failures that cannot be repaired by a tolerance table."""
    for observation in observations:
        for measurement in observation["measurements"]:
            metric = measurement["metric"]
            if _top1(metric) is False:
                return {
                    "verdict": "FAIL",
                    "reason": "top1_agreement",
                    "case_id": observation["case_id"],
                    "step": measurement["step"],
                    "location": measurement["location"],
                    "metric": metric,
                }
    return {
        "verdict": "CANDIDATE_PASS",
        "reason": "thresholds require human promotion before an official PASS",
    }


def raw_measurements_digest(observations: Iterable[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(observations)))


def load_candidate(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "status",
        "margin_multiplier",
        "raw_measurements_sha256",
        "records",
    }
    if set(value) != expected or value["schema_version"] != 1:
        raise CalibrationError("invalid candidate tolerance schema")
    if value["status"] != "candidate_review_required" or value["margin_multiplier"] != 2.0:
        raise CalibrationError("candidate tolerance status changed")
    return value


def _metric_delta(metric: dict[str, Any]) -> float:
    tensor = metric.get("tensor", metric)
    return float(tensor["max_abs_delta"])


def _top1(metric: dict[str, Any]) -> bool | None:
    return metric.get("top1_agreement")
