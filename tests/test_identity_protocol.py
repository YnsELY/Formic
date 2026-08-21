from __future__ import annotations

import pytest

from formic.science.identity.protocol import (
    AlignedProtocol,
    InvalidMeasurement,
    NoisePair,
    PairTrace,
    SharedShapeWarmups,
    shared_noise_floor_schedule,
)
from formic.science.identity.types import InputShape


def test_warmups_capture_nothing_and_last_two_measured_must_be_stable():
    calls: list[tuple[bool, int]] = []

    def run(capture, ordinal):
        calls.append((capture, ordinal))
        return PairTrace("ref", "runner", {"ordinal": ordinal} if capture else None, 2 if capture else 0)

    result = AlignedProtocol(warmup_traces=3, measured_traces=3).run_shape(
        InputShape(1, 17), run
    )
    assert result.stable
    assert calls == [
        (False, 0), (False, 1), (False, 2),
        (True, 3), (True, 4), (True, 5),
    ]


def test_warmup_state_capture_is_rejected():
    def run(capture, ordinal):
        return PairTrace("x", "x", None, 1)

    with pytest.raises(InvalidMeasurement, match="warmup"):
        AlignedProtocol(warmup_traces=1, measured_traces=2).run_shape(
            InputShape(1, 8), run
        )


def test_warmups_are_shared_by_exact_shape_within_one_process_protocol():
    calls: list[tuple[bool, int]] = []

    def run(capture, ordinal):
        calls.append((capture, ordinal))
        return PairTrace("same", "same", {} if capture else None, 0)

    protocol = AlignedProtocol(warmup_traces=3, measured_traces=2)
    shape = InputShape(1, 17)
    first = protocol.run_shape(shape, run)
    second = protocol.run_shape(shape, run)

    assert first.warmup_traces == 3
    assert second.warmup_traces == 0
    assert sum(not capture for capture, _ in calls) == 3


def test_dependent_path_only_schedules_warmups_for_a_missing_shape():
    ledger = SharedShapeWarmups(2)
    prefix = InputShape(1, 8)
    decode = InputShape(1, 1, 8)

    assert ledger.required_path_traces((prefix,)) == 2
    ledger.record_path_trace((prefix,))
    ledger.record_path_trace((prefix,))
    assert ledger.required_path_traces((prefix, decode)) == 2
    ledger.record_path_trace((prefix, decode))
    ledger.record_path_trace((prefix, decode))
    assert ledger.count(prefix) == 2
    assert ledger.count(decode) == 2
    assert ledger.required_path_traces((prefix, decode)) == 0


def test_unstable_last_two_traces_invalidate_measurement():
    def run(capture, ordinal):
        return PairTrace(f"ref-{ordinal}", "runner", {} if capture else None, 0)

    with pytest.raises(InvalidMeasurement, match="not exact"):
        AlignedProtocol(warmup_traces=0, measured_traces=2).run_shape(
            InputShape(1, 8), run
        )


def test_noise_floor_calendar_is_shared_and_complete():
    schedule = shared_noise_floor_schedule(3)
    assert len(schedule) == 9
    assert schedule[:3] == (
        (0, NoisePair.REFERENCE_REFERENCE),
        (0, NoisePair.RUNNER_RUNNER),
        (0, NoisePair.RUNNER_REFERENCE),
    )
    with pytest.raises(ValueError):
        shared_noise_floor_schedule(2)
