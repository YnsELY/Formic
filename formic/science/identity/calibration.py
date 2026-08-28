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
    reference_floor_observations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build a non-governing tolerance candidate from measured observations.

    Exact rows need no human wording.  Bounded rows remain explicitly
    ``REVIEW_REQUIRED``: the campaign records numbers but never assigns a
    causal explanation in the pod session.
    """
    floors = list(reference_floor_observations)
    floor_repetitions = {int(item["repetition"]) for item in floors}
    if len(floor_repetitions) < 3:
        raise CalibrationError("reference floor lacks three measured repetitions")
    if any(item.get("point") != "logits" for item in floors):
        raise CalibrationError("economical reference floor must be logits-only")
    logits_reference_floor = max(float(item["max_abs_delta"]) for item in floors)

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
        reference_floor = logits_reference_floor if point == "logits" else 0.0
        for item in items:
            item["reference_floor"] = reference_floor
        exact = (
            observed_max == 0.0
            and reference_floor == 0.0
            and all(item["top1_agreement"] is not False for item in items)
        )
        records.append(
            {
                "mode": mode,
                "point": point,
                "length_class": length_class,
                "criterion": "exact" if exact else "bounded",
                "max_abs_delta": (
                    0.0
                    if exact
                    else max(2.0 * observed_max, reference_floor)
                ),
                "physical_justification": None if exact else "REVIEW_REQUIRED",
                "observations": items,
                "observed_max_abs_delta": observed_max,
                "reference_floor_max_abs_delta": reference_floor,
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
    """Summarise cross-path top-1 disagreements without failing the run.

    Calibration compares a Formic path with a *different* canonical stock
    path (cached against full recomputation, segmented against full
    prefixes), so its comparisons are cross-position by construction. Run
    a40-2026-08-28-r1 measured top-1 agreement of 3/8, 1/8 and 2/8 on stable
    committed cases while the same-path control (recompute against
    recompute) stayed exact at 8/8 with zero delta: the flips are a property
    of the backend between two execution paths, not evidence against wrapper
    identity, which the aligned exact gates decide. They are therefore
    counted here and remain blocking where the protocol is aligned
    (``verdict.evaluate``). Every affected tolerance row is still forced to
    ``bounded``/``REVIEW_REQUIRED`` by :func:`build_candidate_tolerances`, so
    a human must justify it before promotion.
    """
    by_case: dict[str, int] = defaultdict(int)
    first: dict[str, Any] | None = None
    for observation in observations:
        for measurement in observation["measurements"]:
            metric = measurement["metric"]
            if _top1(metric) is False:
                by_case[observation["case_id"]] += 1
                if first is None:
                    first = {
                        "case_id": observation["case_id"],
                        "step": measurement["step"],
                        "location": measurement["location"],
                        "metric": metric,
                    }
    return {
        "verdict": "CANDIDATE_PASS",
        "reason": "thresholds require human promotion before an official PASS",
        "top1_disagreements": {
            "total": sum(by_case.values()),
            "is_blocking": False,
            "by_case": dict(sorted(by_case.items())),
            "first": first,
        },
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
