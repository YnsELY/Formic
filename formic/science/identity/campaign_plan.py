"""Immutable execution plan for the final SPEC-02 A40 campaign.

The plan contains no model code.  Keeping it pure makes the expensive GPU
session auditable from weight-free tests: each phase, prompt, execution mode,
and expected forward count is fixed before the checkpoint is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from formic.config.schema import RunConfig
from formic.science.identity.budget import EXPECTED_PHASE_FORWARDS, PREFLIGHT_FORWARDS
from formic.science.identity.prompts import FrozenPrompt, FrozenPromptCorpus
from formic.science.identity.types import ExecutionMode


PHASE_ORDER = (
    "preflight",
    "trace_inertness",
    "legacy_continuity",
    "noise_floor",
    "snapshot_restore",
    "reference_continuations",
    "short",
    "medium",
    "long",
    "accumulation_probe_64",
)


@dataclass(frozen=True)
class CampaignPath:
    """One complete forward path, not an individual model forward."""

    prompt: FrozenPrompt
    mode: ExecutionMode
    segmentation: str | None = None

    @property
    def key(self) -> str:
        return "__".join(
            (self.prompt.id, self.mode.value, self.segmentation or "none")
        )


@dataclass(frozen=True)
class CampaignPlan:
    """Fully pinned plan used by the campaign runner and its preflight."""

    corpus_sha256: str
    preflight_paths: tuple[CampaignPath, ...]
    calibration_paths: tuple[CampaignPath, ...]
    phase_forwards: dict[str, int]

    @property
    def total_forwards(self) -> int:
        return PREFLIGHT_FORWARDS + sum(self.phase_forwards.values())

    def validate(self) -> None:
        if tuple(self.phase_forwards) != PHASE_ORDER[1:]:
            raise ValueError("SPEC-02 campaign phase order changed")
        if self.phase_forwards != EXPECTED_PHASE_FORWARDS:
            raise ValueError("SPEC-02 campaign forward budget changed")
        if len(self.preflight_paths) != 18:
            raise ValueError("preflight must contain 18 timed paths")
        if self.total_forwards != 4_139:
            raise ValueError(f"campaign forwards {self.total_forwards} != 4139")


def build_campaign_plan(config: RunConfig, corpus: FrozenPromptCorpus) -> CampaignPlan:
    """Build the validated Option-B/horizon-8 plan from frozen inputs only."""
    config.validate()
    corpus.validate()
    by_id = {prompt.id: prompt for prompt in corpus.prompts}
    selected = tuple(by_id[item] for item in config.identity.decode_prompt_ids)
    if tuple(prompt.length_class for prompt in selected) != ("short", "medium", "long"):
        raise ValueError("decode prompts must remain ordered short/medium/long")

    preflight: list[CampaignPath] = []
    calibration: list[CampaignPath] = []
    for prompt in selected:
        segmentations = (
            config.identity.long_calibration_segmentations
            if prompt.length_class == "long"
            else config.identity.calibration_segmentations
        )
        preflight.append(CampaignPath(prompt, ExecutionMode.PREFILL_FULL))
        preflight.extend(
            CampaignPath(prompt, ExecutionMode.PREFILL_SEGMENTED, segmentation)
            for segmentation in segmentations
        )
        preflight.append(CampaignPath(prompt, ExecutionMode.DECODE_CACHED))
        if prompt.length_class in config.identity.recompute_classes:
            preflight.append(CampaignPath(prompt, ExecutionMode.DECODE_RECOMPUTE))

    for length_class in ("short", "medium", "long"):
        class_prompts = tuple(
            prompt
            for prompt in corpus.prompts
            if prompt.set_name == "calibration" and prompt.length_class == length_class
        )
        segmentations = (
            config.identity.long_calibration_segmentations
            if length_class == "long"
            else config.identity.calibration_segmentations
        )
        for prompt in class_prompts:
            calibration.append(CampaignPath(prompt, ExecutionMode.PREFILL_FULL))
            calibration.extend(
                CampaignPath(prompt, ExecutionMode.PREFILL_SEGMENTED, segmentation)
                for segmentation in segmentations
            )
        decode_prompt = by_id[config.identity.decode_prompt_ids[("short", "medium", "long").index(length_class)]]
        calibration.append(CampaignPath(decode_prompt, ExecutionMode.DECODE_CACHED))
        if length_class in config.identity.recompute_classes:
            calibration.append(CampaignPath(decode_prompt, ExecutionMode.DECODE_RECOMPUTE))

    plan = CampaignPlan(
        corpus_sha256=corpus.corpus_sha256,
        preflight_paths=tuple(preflight),
        calibration_paths=tuple(calibration),
        phase_forwards=dict(EXPECTED_PHASE_FORWARDS),
    )
    plan.validate()
    return plan


def timing_continuation(prompt: FrozenPrompt, steps: int) -> tuple[int, ...]:
    """Deterministic non-measurement continuation approved for preflight.

    The final token already belongs to the frozen prompt and is a valid token
    ID.  Its repetition determines cache/input shapes without selecting a
    token from logits or consuming a random seed.
    """
    if steps <= 0:
        raise ValueError("timing continuation requires positive steps")
    return (prompt.token_ids[-1],) * steps


def paths_for_class(
    plan: CampaignPlan, length_class: Literal["short", "medium", "long"]
) -> tuple[CampaignPath, ...]:
    return tuple(item for item in plan.calibration_paths if item.prompt.length_class == length_class)
