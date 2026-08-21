"""Value types shared by identity measurements and verdict artefacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal


class ExecutionMode(str, Enum):
    PREFILL_FULL = "prefill_full"
    PREFILL_SEGMENTED = "prefill_segmented"
    DECODE_CACHED = "decode_cached"
    DECODE_RECOMPUTE = "decode_recompute"


class SamplingMode(str, Enum):
    GREEDY = "greedy"
    SEEDED_SAMPLING = "seeded_sampling"


class ComparisonPoint(str, Enum):
    LOGITS = "logits"
    HIDDEN_STATE = "hidden_state"
    GDN_STATE = "gdn_state"
    ATTENTION_KV = "attention_kv"
    MODEL_STATE = "model_state"


class CaptureProfile(str, Enum):
    FULL_BOUNDARIES = "full_boundaries"
    FINAL_STATE_ONLY = "final_state_only"
    LOGITS_ONLY = "logits_only"


Applicability = Literal["applicable", "not_applicable"]


@dataclass(frozen=True, order=True)
class InputShape:
    """Kernel-relevant input shape; warmups are keyed by exact length."""

    batch_size: int
    input_length: int
    cached_length: int = 0

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("SPEC-02 requires batch size 1 (A8)")
        if self.input_length <= 0 or self.cached_length < 0:
            raise ValueError("invalid input shape")

    @property
    def key(self) -> str:
        return f"b1-i{self.input_length}-c{self.cached_length}"


@dataclass(frozen=True)
class ComparisonLocation:
    point: ComparisonPoint
    boundary: str | None = None
    layer: int | None = None
    component: str | None = None

    @property
    def key(self) -> str:
        parts = [self.point.value]
        if self.boundary is not None:
            parts.append(self.boundary)
        if self.layer is not None:
            parts.append(f"layer_{self.layer}")
        if self.component is not None:
            parts.append(self.component)
        return "/".join(parts)


@dataclass(frozen=True)
class FirstDivergence:
    step: int
    location: ComparisonLocation
    coordinate: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["location"]["point"] = self.location.point.value
        value["coordinate"] = list(self.coordinate) if self.coordinate is not None else None
        return value


@dataclass(frozen=True)
class CaseKey:
    prompt_id: str
    length_class: Literal["short", "medium", "long"]
    exact_prompt_length: int
    mode: ExecutionMode
    sampling: SamplingMode
    segmentation: str | None
    continuation_seed: int | None
    repetition: int

    def __post_init__(self) -> None:
        if self.exact_prompt_length <= 0:
            raise ValueError("exact_prompt_length must be positive")
        if self.continuation_seed is not None and self.continuation_seed < 0:
            raise ValueError("continuation seed must be non-negative")
        if self.repetition < 0:
            raise ValueError("repetition must be non-negative")
        if self.mode is ExecutionMode.PREFILL_SEGMENTED and self.segmentation is None:
            raise ValueError("segmented prefill requires a segmentation")
        if self.mode is not ExecutionMode.PREFILL_SEGMENTED and self.segmentation is not None:
            raise ValueError("segmentation only applies to segmented prefill")

    @property
    def stable_id(self) -> str:
        segmentation = self.segmentation or "none"
        return (
            f"{self.prompt_id}__{self.mode.value}__{self.sampling.value}__"
            f"{segmentation}__cs{self.continuation_seed}__r{self.repetition}"
        )
