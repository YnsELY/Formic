from __future__ import annotations

import torch

from formic.science.identity.metrics import compare_logits, compare_tensors
from formic.science.identity.tolerances import (
    Observation,
    ToleranceCatalogue,
    ToleranceEvidence,
    ToleranceRecord,
)
from formic.science.identity.types import (
    ComparisonLocation,
    ComparisonPoint,
    ExecutionMode,
    InputShape,
)
from formic.science.identity.verdict import MeasuredComparison, evaluate


def _evidence(delta):
    return ToleranceEvidence(
        exact_lengths=(16,),
        observations=tuple(
            Observation(0, repetition, delta, 0.0)
            for repetition in (0, 1, 2)
        ),
        measurement_artifact="artifact.json",
        measurement_artifact_sha256="a" * 64,
    )


def _catalogue(point, *, bounded=False):
    record = ToleranceRecord(
        ExecutionMode.DECODE_CACHED,
        point,
        "short",
        "bounded" if bounded else "exact",
        0.5 if bounded else 0.0,
        "measured rounding" if bounded else None,
        _evidence(0.25 if bounded else 0.0),
    )
    result = ToleranceCatalogue(1, "calibrated", 2.0, (record,), "f" * 64)
    result.validate()
    return result


def test_pass_uses_exact_default():
    metric = compare_tensors(torch.tensor([1.0]), torch.tensor([1.0]))
    measurement = MeasuredComparison(
        0,
        ExecutionMode.DECODE_CACHED,
        "short",
        InputShape(1, 1, 8),
        ComparisonLocation(ComparisonPoint.HIDDEN_STATE, "G1_G2"),
        metric,
    )
    assert evaluate((measurement,), _catalogue(ComparisonPoint.HIDDEN_STATE)).passed


def test_failure_reports_step_location_coordinate_and_threshold():
    metric = compare_tensors(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.75]]))
    measurement = MeasuredComparison(
        4,
        ExecutionMode.DECODE_CACHED,
        "short",
        InputShape(1, 1, 12),
        ComparisonLocation(ComparisonPoint.GDN_STATE, "G2_G3", 5, "recurrent"),
        metric,
    )
    verdict = evaluate((measurement,), _catalogue(ComparisonPoint.GDN_STATE, bounded=True))
    assert not verdict.passed
    assert verdict.first_divergence.step == 4
    assert verdict.first_divergence.location.layer == 5
    assert verdict.first_divergence.coordinate == (0, 1)
    assert verdict.violated_metric == "max_abs_delta"
    assert verdict.observed_value == 0.75
    assert verdict.tolerance == 0.5


def test_top1_is_blocking_while_kl_is_only_diagnostic():
    metric = compare_logits(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0]))
    measurement = MeasuredComparison(
        2,
        ExecutionMode.DECODE_CACHED,
        "short",
        InputShape(1, 1, 10),
        ComparisonLocation(ComparisonPoint.LOGITS),
        metric,
    )
    verdict = evaluate((measurement,), _catalogue(ComparisonPoint.LOGITS, bounded=True))
    assert not verdict.passed
    assert verdict.violated_metric == "top1_agreement"
    assert verdict.diagnostic_kl > 0
