"""Measured, capture-free timing preflight for the final SPEC-02 session."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from formic.backbone.loader import BackboneHandle
from formic.science.identity.artifacts import atomic_write_json
from formic.science.identity.budget import EXPECTED_PHASE_FORWARDS, PhaseEstimate, PreflightEstimate
from formic.science.identity.campaign_plan import CampaignPath, CampaignPlan, timing_continuation
from formic.science.identity.executor import (
    Endpoint,
    execute_path,
    execute_reference_for_candidate,
)


# Volumes from the approved cost plan (v3, burn-in included; regenerate with
# scripts/estimate_step2_campaign.py).  They are estimates only; the exact
# captured-state accounting remains in every measured case artefact.
_PHASE_TRANSFER_GIB = {
    "trace_inertness": 4.22,
    "legacy_continuity": 1.57,
    "noise_floor": 0.26,
    "snapshot_restore": 4.81,
    "reference_continuations": 0.0,
    "short": 18.93,
    "medium": 35.23,
    "long": 18.87,
    "accumulation_probe_64": 0.36,
}


@dataclass(frozen=True)
class PathTiming:
    path: CampaignPath
    dry_seconds: float
    measured_seconds: tuple[float, float]
    reference_dry_seconds: float | None = None
    reference_measured_seconds: tuple[float, float] | None = None

    @property
    def conservative_seconds(self) -> float:
        return max(self.measured_seconds)

    @property
    def reference_conservative_seconds(self) -> float:
        measured = self.reference_measured_seconds or self.measured_seconds
        return max(measured)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path.key,
            "prompt_id": self.path.prompt.id,
            "length_class": self.path.prompt.length_class,
            "mode": self.path.mode.value,
            "segmentation": self.path.segmentation,
            "dry_seconds": self.dry_seconds,
            "measured_seconds": list(self.measured_seconds),
            "conservative_seconds": self.conservative_seconds,
            "reference_dry_seconds": (
                self.dry_seconds
                if self.reference_dry_seconds is None
                else self.reference_dry_seconds
            ),
            "reference_measured_seconds": list(
                self.reference_measured_seconds or self.measured_seconds
            ),
            "reference_conservative_seconds": self.reference_conservative_seconds,
            "reference_reuses_candidate_timing": self.reference_measured_seconds is None,
        }


@dataclass(frozen=True)
class PreflightRun:
    estimate: PreflightEstimate
    timings: tuple[PathTiming, ...]
    transfer_bytes_per_second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "timings": [item.to_dict() for item in self.timings],
            "transfer_bytes_per_second": self.transfer_bytes_per_second,
            "estimate": {
                "schema_version": self.estimate.schema_version,
                "protocol": self.estimate.protocol,
                "model_processes": self.estimate.model_processes,
                "model_load_seconds": self.estimate.model_load_seconds,
                "preflight_forwards": self.estimate.preflight_forwards,
                "preflight_elapsed_seconds": self.estimate.preflight_elapsed_seconds,
                "phases": [
                    {
                        "name": phase.name,
                        "forwards": phase.forwards,
                        "estimated_seconds": phase.estimated_seconds,
                    }
                    for phase in self.estimate.phases
                ],
            },
        }


def run_preflight(
    handle: BackboneHandle,
    plan: CampaignPlan,
    *,
    estimate_path: str | Path,
    details_path: str | Path,
    memory_observer: Callable[[str], None] | None = None,
) -> PreflightRun:
    """Time the 18 candidates and distinct canonical companions without capture.

    Each execution is run once dry and twice timed. Identical reference and
    candidate paths reuse one timing; segmented-prefix and cached/recompute
    comparisons are timed separately. The slower execution drives the report.
    """
    plan.validate()
    endpoint = Endpoint("reference", handle.model, handle.view, False)
    started = time.perf_counter()
    timings: list[PathTiming] = []
    for path in plan.preflight_paths:
        dry = _time_path(endpoint, path, handle.config.identity.decode_tokens)
        timed = (
            _time_path(endpoint, path, handle.config.identity.decode_tokens),
            _time_path(endpoint, path, handle.config.identity.decode_tokens),
        )
        if _reference_reuses_candidate_timing(path):
            reference_dry = None
            reference_timed = None
        else:
            reference_dry = _time_reference_path(
                endpoint, path, handle.config.identity.decode_tokens
            )
            reference_timed = (
                _time_reference_path(
                    endpoint, path, handle.config.identity.decode_tokens
                ),
                _time_reference_path(
                    endpoint, path, handle.config.identity.decode_tokens
                ),
            )
        timings.append(
            PathTiming(path, dry, timed, reference_dry, reference_timed)
        )
        # Release transient workspaces only after the complete path.  Doing it
        # between measured repetitions would perturb the allocation history
        # that this protocol intentionally observes.
        release_cuda_working_set()
        if memory_observer is not None:
            memory_observer(f"after_preflight_path:{path.key}")
    transfer_rate = _measure_transfer_rate(handle.device)
    release_cuda_working_set()
    if memory_observer is not None:
        memory_observer("after_preflight_transfer_control")
    elapsed = time.perf_counter() - started
    estimate = _estimate_from_timings(
        timings,
        model_load_seconds=handle.load_seconds,
        preflight_elapsed_seconds=elapsed,
        transfer_bytes_per_second=transfer_rate,
    )
    estimate.validate()
    atomic_write_json(estimate_path, _estimate_json(estimate))
    result = PreflightRun(estimate, tuple(timings), transfer_rate)
    atomic_write_json(details_path, result.to_dict())
    return result


def release_cuda_working_set() -> None:
    """Collect transient tensors and return inactive CUDA blocks to the driver.

    Live model parameters and execution state are not touched.  This helper is
    used only at path/phase boundaries, never between measured repetitions.
    """
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _time_path(endpoint: Endpoint, path: CampaignPath, decode_steps: int) -> float:
    _synchronise()
    started = time.perf_counter()
    execute_path(
        endpoint,
        prompt_token_ids=path.prompt.token_ids,
        mode=path.mode,
        length_class=path.prompt.length_class,
        segmentation=path.segmentation,
        forced_token_ids=(
            timing_continuation(path.prompt, decode_steps)
            if path.mode.value.startswith("decode_")
            else ()
        ),
        capture=False,
    )
    _synchronise()
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("preflight clock produced a non-positive duration")
    return elapsed


def _reference_reuses_candidate_timing(path: CampaignPath) -> bool:
    return (
        path.mode.value in ("prefill_full", "decode_recompute")
        or (path.mode.value == "decode_cached" and path.prompt.length_class == "long")
    )


def _time_reference_path(
    endpoint: Endpoint, path: CampaignPath, decode_steps: int
) -> float:
    _synchronise()
    started = time.perf_counter()
    execute_reference_for_candidate(
        endpoint,
        prompt_token_ids=path.prompt.token_ids,
        candidate_mode=path.mode,
        length_class=path.prompt.length_class,
        segmentation=path.segmentation,
        forced_token_ids=(
            timing_continuation(path.prompt, decode_steps)
            if path.mode.value.startswith("decode_")
            else ()
        ),
        capture=False,
    )
    _synchronise()
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("reference preflight clock produced a non-positive duration")
    return elapsed


def _synchronise() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_transfer_rate(device: Any) -> float:
    """Measure a representative GPU-to-CPU copy without identity state."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SPEC-02 preflight requires CUDA")
    # With the audited auto device map the first parameter may reside on CPU
    # while active decoder layers are offloaded to CUDA.  The transfer control
    # still belongs on the sole visible campaign GPU.
    if getattr(device, "type", None) != "cuda":
        device = torch.device("cuda", 0)
    size = 16 * 1024 * 1024
    source = torch.empty(size, device=device, dtype=torch.uint8)
    durations: list[float] = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        destination = source.to(device="cpu", copy=True)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        del destination
    del source
    elapsed = max(durations)
    if elapsed <= 0:
        raise RuntimeError("GPU-to-CPU timing produced a non-positive duration")
    return size / elapsed


def _estimate_from_timings(
    timings: list[PathTiming],
    *,
    model_load_seconds: float,
    preflight_elapsed_seconds: float,
    transfer_bytes_per_second: float,
) -> PreflightEstimate:
    if model_load_seconds <= 0 or preflight_elapsed_seconds <= 0 or transfer_bytes_per_second <= 0:
        raise ValueError("preflight measurements must be positive")
    lookup = {
        (item.path.prompt.length_class, item.path.mode.value, item.path.segmentation): item.conservative_seconds
        for item in timings
    }
    reference_lookup = {
        (item.path.prompt.length_class, item.path.mode.value, item.path.segmentation): item.reference_conservative_seconds
        for item in timings
    }

    def path_time(length_class: str, mode: str, segmentation: str | None = None) -> float:
        try:
            return lookup[(length_class, mode, segmentation)]
        except KeyError as exc:
            raise RuntimeError(f"preflight did not time {length_class}/{mode}/{segmentation}") from exc

    def reference_time(
        length_class: str, mode: str, segmentation: str | None = None
    ) -> float:
        try:
            return reference_lookup[(length_class, mode, segmentation)]
        except KeyError as exc:
            raise RuntimeError(
                f"preflight did not time canonical reference {length_class}/{mode}/{segmentation}"
            ) from exc

    def with_transfer(name: str, execution_seconds: float) -> float:
        return execution_seconds + (_PHASE_TRANSFER_GIB[name] * 2**30 / transfer_bytes_per_second)

    # Path-trace multipliers follow the v3 protocol: per-endpoint warmups plus
    # measured-then-discarded burn-in after every non-empty warmup block.  See
    # scripts/estimate_step2_campaign.py for the per-phase derivation.
    phase_seconds: dict[str, float] = {}
    phase_seconds["trace_inertness"] = with_transfer(
        "trace_inertness",
        96 * path_time("short", "prefill_full")
        + 24 * path_time("medium", "prefill_full")
        + 24 * path_time("long", "prefill_full"),
    )
    phase_seconds["legacy_continuity"] = with_transfer(
        "legacy_continuity", 484 * path_time("short", "decode_cached")
    )
    phase_seconds["noise_floor"] = with_transfer(
        "noise_floor",
        56 * path_time("short", "decode_cached")
        + 38 * path_time("medium", "decode_cached"),
    )
    phase_seconds["snapshot_restore"] = with_transfer(
        "snapshot_restore", 8 * path_time("short", "decode_cached")
    )
    phase_seconds["reference_continuations"] = with_transfer(
        "reference_continuations",
        4 * (
            path_time("short", "decode_cached")
            + path_time("medium", "decode_cached")
            + path_time("long", "decode_cached")
        ),
    )
    for length_class in ("short", "medium", "long"):
        segmentations = ("median", "quarters") if length_class == "long" else (
            "early", "median", "late", "quarters"
        )
        prefill_paths = 40 * path_time(length_class, "prefill_full") + 20 * sum(
            path_time(length_class, "prefill_segmented", item)
            + reference_time(length_class, "prefill_segmented", item)
            for item in segmentations
        )
        if length_class == "long":
            # Long cached decode warms both endpoints (6+6), burns in one
            # pair (1+1) and measures six pairs: 13 traces per side.
            decode_paths = 26 * path_time(length_class, "decode_cached")
        else:
            # The cached candidate is calibrated against a full-recompute
            # reference: 13 cached traces (6 warm + 1 burn-in + 6 measured)
            # + 7 recompute reference traces (1 burn-in + 6 measured), in
            # addition to the 26 recompute/recompute path traces.
            decode_paths = (
                13 * path_time(length_class, "decode_cached")
                + 7 * reference_time(length_class, "decode_cached")
                + 26 * path_time(length_class, "decode_recompute")
            )
        phase_seconds[length_class] = with_transfer(
            length_class,
            prefill_paths + decode_paths,
        )
    phase_seconds["accumulation_probe_64"] = with_transfer(
        "accumulation_probe_64",
        576 * (path_time("short", "decode_cached") / 8)
        + 576 * (reference_time("short", "decode_cached") / 8)
        + 576 * (path_time("medium", "decode_cached") / 8)
        + 576 * (reference_time("medium", "decode_cached") / 8),
    )
    phases = tuple(
        PhaseEstimate(name, EXPECTED_PHASE_FORWARDS[name], phase_seconds[name])
        for name in EXPECTED_PHASE_FORWARDS
    )
    from formic.science.identity.budget import PREFLIGHT_FORWARDS, PROTOCOL_ID

    return PreflightEstimate(
        1,
        PROTOCOL_ID,
        1,
        model_load_seconds,
        PREFLIGHT_FORWARDS,
        preflight_elapsed_seconds,
        phases,
    )


def _estimate_json(value: PreflightEstimate) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "protocol": value.protocol,
        "model_processes": value.model_processes,
        "model_load_seconds": value.model_load_seconds,
        "preflight_forwards": value.preflight_forwards,
        "preflight_elapsed_seconds": value.preflight_elapsed_seconds,
        "phases": [
            {
                "name": phase.name,
                "forwards": phase.forwards,
                "estimated_seconds": phase.estimated_seconds,
            }
            for phase in value.phases
        ],
    }
