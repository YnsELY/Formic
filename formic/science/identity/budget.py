"""Non-blocking A40 duration estimator for the approved SPEC-02 plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "SPEC-02-h8-option-b-balanced-v3"
PREFLIGHT_FORWARDS = 333
# v3 adds measured-then-discarded burn-in after every non-empty warmup block
# (4 pair traces for the paired gates, 1 repetition for cross-path cases),
# per-endpoint warmups (the runner is now warmed for prefill_full,
# decode_recompute and long cached decode), and three measured repetitions for
# the 64-frame probe like every other measurement.  See ADR-0005 and
# scripts/estimate_step2_campaign.py for the per-phase derivation.
EXPECTED_PHASE_FORWARDS = {
    "trace_inertness": 144,
    "legacy_continuity": 3_872,
    "noise_floor": 752,
    "snapshot_restore": 64,
    "reference_continuations": 96,
    "short": 808,
    "medium": 808,
    "long": 488,
    "accumulation_probe_64": 2_560,
}


class EstimateError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhaseEstimate:
    name: str
    forwards: int
    estimated_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "forwards": self.forwards,
            "estimated_seconds": self.estimated_seconds,
            "estimated_hours": self.estimated_seconds / 3_600,
        }


@dataclass(frozen=True)
class PreflightEstimate:
    schema_version: int
    protocol: str
    model_processes: int
    model_load_seconds: float
    preflight_forwards: int
    preflight_elapsed_seconds: float
    phases: tuple[PhaseEstimate, ...]

    def validate(self) -> None:
        if self.schema_version != 1 or self.protocol != PROTOCOL_ID:
            raise EstimateError("preflight protocol/schema does not match horizon-8 plan")
        if self.model_processes != 1:
            raise EstimateError("SPEC-02 session is planned as one model process")
        if self.model_load_seconds <= 0:
            raise EstimateError("model load time must be measured and positive")
        if self.preflight_forwards != PREFLIGHT_FORWARDS:
            raise EstimateError("preflight forward count changed")
        if self.preflight_elapsed_seconds <= 0:
            raise EstimateError("preflight elapsed time must be measured and positive")
        actual = {phase.name: phase.forwards for phase in self.phases}
        if actual != EXPECTED_PHASE_FORWARDS:
            raise EstimateError(
                f"phase forward plan changed: expected={EXPECTED_PHASE_FORWARDS}, actual={actual}"
            )
        if any(phase.estimated_seconds <= 0 for phase in self.phases):
            raise EstimateError("every phase requires a positive preflight estimate")


@dataclass(frozen=True)
class EstimateReport:
    protocol: str
    model_processes: int
    model_load_seconds: float
    preflight_elapsed_seconds: float
    remaining_estimated_seconds: float
    total_estimated_seconds: float
    total_forwards: int
    phases: tuple[PhaseEstimate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "report": "ESTIMATE",
            "protocol": self.protocol,
            "model_processes": self.model_processes,
            "model_load_seconds": self.model_load_seconds,
            "model_load_hours": self.model_load_seconds / 3_600,
            "preflight_elapsed_seconds": self.preflight_elapsed_seconds,
            "preflight_elapsed_hours": self.preflight_elapsed_seconds / 3_600,
            "remaining_estimated_seconds": self.remaining_estimated_seconds,
            "remaining_estimated_hours": self.remaining_estimated_seconds / 3_600,
            "total_estimated_seconds": self.total_estimated_seconds,
            "total_estimated_hours": self.total_estimated_seconds / 3_600,
            "total_forwards": self.total_forwards,
            "phases": [phase.to_dict() for phase in self.phases],
        }


def load_preflight_estimate(path: str | Path) -> PreflightEstimate:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _strict(
        value,
        {
            "schema_version",
            "protocol",
            "model_processes",
            "model_load_seconds",
            "preflight_forwards",
            "preflight_elapsed_seconds",
            "phases",
        },
        "preflight",
    )
    if not isinstance(value["phases"], list):
        raise EstimateError("preflight.phases must be a list")
    phases = []
    for index, item in enumerate(value["phases"]):
        _strict(item, {"name", "forwards", "estimated_seconds"}, f"phases[{index}]")
        phases.append(PhaseEstimate(item["name"], item["forwards"], item["estimated_seconds"]))
    result = PreflightEstimate(
        value["schema_version"],
        value["protocol"],
        value["model_processes"],
        value["model_load_seconds"],
        value["preflight_forwards"],
        value["preflight_elapsed_seconds"],
        tuple(phases),
    )
    result.validate()
    return result


def report_estimate(preflight: PreflightEstimate) -> EstimateReport:
    preflight.validate()
    remaining = sum(phase.estimated_seconds for phase in preflight.phases)
    return EstimateReport(
        PROTOCOL_ID,
        preflight.model_processes,
        preflight.model_load_seconds,
        preflight.preflight_elapsed_seconds,
        remaining,
        preflight.preflight_elapsed_seconds + remaining,
        PREFLIGHT_FORWARDS + sum(EXPECTED_PHASE_FORWARDS.values()),
        preflight.phases,
    )


def _strict(value: Any, expected: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise EstimateError(f"{path} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise EstimateError(f"{path}: missing={missing}, unknown={unknown}")
