"""Blocking verdict evaluation and actionable first-divergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from formic.science.identity.metrics import LogitComparison, TensorComparison
from formic.science.identity.tolerances import ToleranceCatalogue
from formic.science.identity.types import (
    ComparisonLocation,
    FirstDivergence,
    InputShape,
    ExecutionMode,
)


@dataclass(frozen=True)
class MeasuredComparison:
    step: int
    mode: ExecutionMode
    length_class: str
    input_shape: InputShape
    location: ComparisonLocation
    metric: TensorComparison | LogitComparison


@dataclass(frozen=True)
class IdentityVerdict:
    passed: bool
    first_divergence: FirstDivergence | None
    violated_metric: str | None
    observed_value: float | bool | None
    tolerance: float | bool | None
    diagnostic_kl: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": "PASS" if self.passed else "FAIL",
            "first_divergence": (
                self.first_divergence.to_dict() if self.first_divergence else None
            ),
            "violated_metric": self.violated_metric,
            "observed_value": self.observed_value,
            "tolerance": self.tolerance,
            "diagnostic_kl": self.diagnostic_kl,
        }


def evaluate(
    measurements: tuple[MeasuredComparison, ...],
    tolerances: ToleranceCatalogue,
) -> IdentityVerdict:
    for measurement in measurements:
        metric = measurement.metric
        tensor = metric.tensor if isinstance(metric, LogitComparison) else metric
        record = tolerances.threshold(
            measurement.mode,
            measurement.location.point,
            measurement.length_class,  # type: ignore[arg-type]
        )
        failed_metric: str | None = None
        observed: float | bool | None = None
        allowed: float | bool | None = None
        if isinstance(metric, LogitComparison) and not metric.top1_agreement:
            failed_metric = "top1_agreement"
            observed = False
            allowed = True
        elif record.criterion == "exact" and not tensor.exact:
            failed_metric = "exact_equality"
            observed = False
            allowed = True
        elif tensor.max_abs_delta > record.max_abs_delta:
            failed_metric = "max_abs_delta"
            observed = tensor.max_abs_delta
            allowed = record.max_abs_delta
        if failed_metric is not None:
            return IdentityVerdict(
                passed=False,
                first_divergence=FirstDivergence(
                    step=measurement.step,
                    location=measurement.location,
                    coordinate=tensor.first_coordinate,
                ),
                violated_metric=failed_metric,
                observed_value=observed,
                tolerance=allowed,
                diagnostic_kl=(
                    metric.kl_next_token if isinstance(metric, LogitComparison) else None
                ),
            )
    return IdentityVerdict(True, None, None, None, None, None)
