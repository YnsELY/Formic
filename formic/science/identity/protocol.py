"""Aligned in-process measurement protocol mandated by ADR-0004/SPEC-02."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from formic.science.identity.types import InputShape


class InvalidMeasurement(RuntimeError):
    """Measured traces were unstable and cannot justify a tolerance or verdict."""


class SharedShapeWarmups:
    """Process-local ledger that schedules warmups only for unseen exact shapes.

    A path may contain several dependent cache shapes.  When one of them still
    needs warming, the complete path is replayed; already-warm prerequisite
    shapes may therefore execute incidentally, but never trigger a new warmup
    schedule on their own.
    """

    def __init__(self, traces_per_shape: int) -> None:
        if traces_per_shape < 0:
            raise ValueError("warmup trace count must be non-negative")
        self.traces_per_shape = traces_per_shape
        self._counts: dict[InputShape, int] = {}

    def required_path_traces(self, shapes: tuple[InputShape, ...]) -> int:
        unique = set(shapes)
        return max(
            (self.traces_per_shape - self._counts.get(shape, 0) for shape in unique),
            default=0,
        )

    def record_path_trace(self, shapes: tuple[InputShape, ...]) -> None:
        for shape in set(shapes):
            self._counts[shape] = min(
                self.traces_per_shape,
                self._counts.get(shape, 0) + 1,
            )

    def count(self, shape: InputShape) -> int:
        return self._counts.get(shape, 0)


class NoisePair(str, Enum):
    REFERENCE_REFERENCE = "reference_reference"
    RUNNER_RUNNER = "runner_runner"
    RUNNER_REFERENCE = "runner_reference"


NOISE_FLOOR_SCHEDULE = (
    NoisePair.REFERENCE_REFERENCE,
    NoisePair.RUNNER_RUNNER,
    NoisePair.RUNNER_REFERENCE,
)


@dataclass(frozen=True)
class PairTrace:
    """Fingerprints for one aligned comparison; payload is measured data only."""

    reference_fingerprint: str
    candidate_fingerprint: str
    payload: Any | None
    captured_state_tensors: int


@dataclass(frozen=True)
class ShapeRun:
    shape: InputShape
    warmup_traces: int
    measured: tuple[PairTrace, ...]
    stable: bool


@dataclass(frozen=True)
class CaseRun:
    shapes: tuple[InputShape, ...]
    warmup_traces: int
    measured: tuple[PairTrace, ...]
    stable: bool


class AlignedProtocol:
    """Warm per exact shape, then retain measurements with a stability assertion."""

    def __init__(
        self,
        *,
        warmup_traces: int,
        measured_traces: int,
        require_last_two_exact: bool = True,
    ) -> None:
        if warmup_traces < 0 or measured_traces < 2:
            raise ValueError("invalid warmup/measured trace counts")
        if not require_last_two_exact:
            raise ValueError("SPEC-02 stability assertion cannot be disabled")
        self.warmup_traces = warmup_traces
        self.measured_traces = measured_traces
        self.shared_warmups = SharedShapeWarmups(warmup_traces)

    def run_shape(
        self,
        shape: InputShape,
        run_pair: Callable[[bool, int], PairTrace],
    ) -> ShapeRun:
        case = self.run_case((shape,), run_pair)
        return ShapeRun(shape, case.warmup_traces, case.measured, case.stable)

    def run_case(
        self,
        shapes: tuple[InputShape, ...],
        run_pair: Callable[[bool, int], PairTrace],
    ) -> CaseRun:
        """Run complete traces where every listed exact shape occurs once."""
        if not shapes:
            raise ValueError("a protocol case must declare its exact shapes")
        scheduled_warmups = self.shared_warmups.required_path_traces(shapes)
        for ordinal in range(scheduled_warmups):
            warmup = run_pair(False, ordinal)
            if warmup.payload is not None or warmup.captured_state_tensors != 0:
                raise InvalidMeasurement(
                    "warmup captured measured payload or state"
                )
            self.shared_warmups.record_path_trace(shapes)

        measured = tuple(
            run_pair(True, self.warmup_traces + ordinal)
            for ordinal in range(self.measured_traces)
        )
        previous, latest = measured[-2:]
        stable = (
            previous.reference_fingerprint == latest.reference_fingerprint
            and previous.candidate_fingerprint == latest.candidate_fingerprint
        )
        if not stable:
            raise InvalidMeasurement(
                "last two measured traces are not exact for shapes "
                + ",".join(shape.key for shape in shapes)
            )
        return CaseRun(shapes, scheduled_warmups, measured, stable)


def shared_noise_floor_schedule(
    repetitions: int,
) -> tuple[tuple[int, NoisePair], ...]:
    """Fixed shared calendar for the economical logits-only noise floor."""
    if repetitions < 3:
        raise ValueError("noise-floor schedule requires >=3 repetitions")
    return tuple(
        (repetition, pair)
        for repetition in range(repetitions)
        for pair in NOISE_FLOOR_SCHEDULE
    )
