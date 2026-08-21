"""Strict, evidence-backed tolerance catalogue for SPEC-02.

No production threshold is defined in code. The default criterion is exact;
bounded criteria exist only as records loaded from the calibrated,
hash-referenced ``tolerances.json`` artefact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from formic.science.identity.types import ComparisonPoint, ExecutionMode

Criterion = Literal["exact", "bounded"]
LengthClass = Literal["short", "medium", "long"]


class ToleranceError(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    continuation_seed: int | None
    repetition: int
    max_abs_delta: float
    reference_floor: float


@dataclass(frozen=True)
class ToleranceEvidence:
    exact_lengths: tuple[int, ...]
    observations: tuple[Observation, ...]
    measurement_artifact: str
    measurement_artifact_sha256: str

    def validate(self) -> None:
        if not self.exact_lengths or any(length <= 0 for length in self.exact_lengths):
            raise ToleranceError("evidence must list positive exact input lengths")
        if len(self.observations) < 3:
            raise ToleranceError("tolerance evidence requires at least 3 repetitions")
        repetitions: set[int] = set()
        for item in self.observations:
            if item.continuation_seed is not None and item.continuation_seed < 0:
                raise ToleranceError("continuation seed must be non-negative")
            if item.repetition < 0:
                raise ToleranceError("repetition must be non-negative")
            if item.max_abs_delta < 0 or item.reference_floor < 0:
                raise ToleranceError("observed deltas must be non-negative")
            repetitions.add(item.repetition)
        if len(repetitions) < 3:
            raise ToleranceError("tolerance evidence requires 3 distinct repetitions")
        if len(self.measurement_artifact_sha256) != 64:
            raise ToleranceError("measurement artefact SHA-256 must contain 64 hex digits")
        try:
            int(self.measurement_artifact_sha256, 16)
        except ValueError as exc:
            raise ToleranceError("invalid measurement artefact SHA-256") from exc

    @property
    def observed_max(self) -> float:
        return max(item.max_abs_delta for item in self.observations)

    @property
    def reference_floor_max(self) -> float:
        return max(item.reference_floor for item in self.observations)


@dataclass(frozen=True)
class ToleranceRecord:
    mode: ExecutionMode
    point: ComparisonPoint
    length_class: LengthClass
    criterion: Criterion
    max_abs_delta: float
    physical_justification: str | None
    evidence: ToleranceEvidence

    @property
    def key(self) -> tuple[str, str, str]:
        return self.mode.value, self.point.value, self.length_class

    def validate(self, margin_multiplier: float) -> None:
        self.evidence.validate()
        if self.length_class not in ("short", "medium", "long"):
            raise ToleranceError(f"invalid length class {self.length_class!r}")
        if self.max_abs_delta < 0:
            raise ToleranceError("max_abs_delta must be non-negative")
        if self.criterion == "exact":
            if self.max_abs_delta != 0:
                raise ToleranceError("exact criteria must have a zero delta threshold")
            if self.evidence.observed_max != 0 or self.evidence.reference_floor_max != 0:
                raise ToleranceError("exact criteria require all observations to be exact")
            if self.physical_justification is not None:
                raise ToleranceError("exact criteria do not take a physical justification")
            return
        if self.criterion != "bounded":
            raise ToleranceError(f"unknown criterion {self.criterion!r}")
        if not self.physical_justification:
            raise ToleranceError("bounded criteria require a named physical justification")
        required = max(
            self.evidence.observed_max * margin_multiplier,
            self.evidence.reference_floor_max,
        )
        if self.max_abs_delta != required:
            raise ToleranceError(
                f"bounded threshold {self.max_abs_delta} != required {required}"
            )


@dataclass(frozen=True)
class ToleranceCatalogue:
    schema_version: int
    status: Literal["calibrated"]
    margin_multiplier: float
    records: tuple[ToleranceRecord, ...]
    source_sha256: str

    def validate(self) -> None:
        if self.schema_version != 1 or self.status != "calibrated":
            raise ToleranceError("only calibrated tolerance schema version 1 is accepted")
        if self.margin_multiplier != 2.0:
            raise ToleranceError("SPEC-02 tolerance margin must equal 2.0")
        if not self.records:
            raise ToleranceError("calibrated catalogue must contain records")
        keys = [record.key for record in self.records]
        if len(keys) != len(set(keys)):
            raise ToleranceError("duplicate tolerance key")
        for record in self.records:
            record.validate(self.margin_multiplier)

    def threshold(
        self, mode: ExecutionMode, point: ComparisonPoint, length_class: LengthClass
    ) -> ToleranceRecord:
        key = mode.value, point.value, length_class
        for record in self.records:
            if record.key == key:
                return record
        raise ToleranceError(f"no calibrated tolerance for {key}")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tolerances(path: str | Path) -> ToleranceCatalogue:
    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ToleranceError(f"invalid tolerance JSON: {source}") from exc
    allowed_root = {"schema_version", "status", "margin_multiplier", "records"}
    _strict_keys(value, allowed_root, "tolerances")
    records = tuple(_record(item, index) for index, item in enumerate(value["records"]))
    catalogue = ToleranceCatalogue(
        schema_version=value["schema_version"],
        status=value["status"],
        margin_multiplier=float(value["margin_multiplier"]),
        records=records,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    catalogue.validate()
    return catalogue


def _record(value: dict[str, Any], index: int) -> ToleranceRecord:
    path = f"records[{index}]"
    _strict_keys(
        value,
        {
            "mode", "point", "length_class", "criterion", "max_abs_delta",
            "physical_justification", "evidence",
        },
        path,
    )
    evidence = value["evidence"]
    _strict_keys(
        evidence,
        {
            "exact_lengths", "observations", "measurement_artifact",
            "measurement_artifact_sha256",
        },
        f"{path}.evidence",
    )
    observations: list[Observation] = []
    for obs_index, item in enumerate(evidence["observations"]):
        _strict_keys(
            item,
            {"continuation_seed", "repetition", "max_abs_delta", "reference_floor"},
            f"{path}.evidence.observations[{obs_index}]",
        )
        observations.append(
            Observation(
                continuation_seed=(
                    None
                    if item["continuation_seed"] is None
                    else int(item["continuation_seed"])
                ),
                repetition=int(item["repetition"]),
                max_abs_delta=float(item["max_abs_delta"]),
                reference_floor=float(item["reference_floor"]),
            )
        )
    try:
        mode = ExecutionMode(value["mode"])
        point = ComparisonPoint(value["point"])
    except ValueError as exc:
        raise ToleranceError(f"{path}: invalid mode or comparison point") from exc
    return ToleranceRecord(
        mode=mode,
        point=point,
        length_class=value["length_class"],
        criterion=value["criterion"],
        max_abs_delta=float(value["max_abs_delta"]),
        physical_justification=value["physical_justification"],
        evidence=ToleranceEvidence(
            exact_lengths=tuple(int(length) for length in evidence["exact_lengths"]),
            observations=tuple(observations),
            measurement_artifact=evidence["measurement_artifact"],
            measurement_artifact_sha256=evidence["measurement_artifact_sha256"],
        ),
    )


def _strict_keys(value: Any, allowed: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise ToleranceError(f"{path} must be an object")
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise ToleranceError(f"{path}: missing={missing}, unknown={unknown}")
