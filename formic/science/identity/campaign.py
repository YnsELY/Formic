"""One-process orchestration of the measured SPEC-02 A40 calibration.

This module deliberately separates the *first* calibration from the later
blocking gate.  A calibration records raw evidence and a review-required
tolerance candidate; it never invents an official PASS before a human promotes
the measured table and its governance evidence.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import torch

from formic.backbone.loader import BackboneHandle, load_backbone
from formic.config.schema import RunConfig
from formic.science.backbone_hash import (
    BackboneHash,
    canonical_backbone_hash,
    load_reusable_backbone_hash,
)
from formic.science.determinism import environment_report, git_commit, git_dirty
from formic.science.identity.artifacts import (
    CampaignIdentity,
    IncrementalCampaignWriter,
    atomic_write_json,
    sha256_bytes,
)
from formic.science.identity.budget import load_preflight_estimate, report_estimate
from formic.science.identity.balanced_gate import (
    run_alternating_noise_floor,
    run_balanced_logits_gate,
)
from formic.science.identity.calibration import (
    build_candidate_tolerances,
    candidate_verdict,
)
from formic.science.identity.campaign_plan import CampaignPath, CampaignPlan, build_campaign_plan, timing_continuation
from formic.science.identity.comparison import TraceComparison
from formic.science.identity.continuation import generate_forced_continuation
from formic.science.identity.executor import (
    AlignedCasePayload,
    Endpoint,
    execute_path,
    execute_reference_for_candidate,
    expected_shapes,
    reference_shapes_for_candidate,
    run_aligned_pair,
    run_cross_path_pair,
    run_greedy_pair,
)
from formic.science.identity.metrics import compare_logits
from formic.science.identity.memory import IncrementalMemoryWriter
from formic.science.identity.preflight import release_cuda_working_set, run_preflight
from formic.science.identity.prompts import FrozenPrompt, FrozenPromptCorpus, load_frozen_corpus
from formic.science.identity.protocol import InvalidMeasurement, SharedShapeWarmups
from formic.science.identity.trace import IdentityTraceCollector
from formic.science.identity.types import CaptureProfile, ExecutionMode, SamplingMode
from formic.state.snapshot import (
    ExecutionStateController,
    PositionState,
    iter_snapshot_tensors,
    snapshot,
    snapshot_fingerprint,
    tensor_storage_identity,
)


class CampaignError(RuntimeError):
    """A campaign precondition or hard gate failed."""


@dataclass(frozen=True)
class CampaignResult:
    completed: bool
    run_root: Path
    message: str
    candidate_verdict: dict[str, Any] | None


def run_gpu_campaign(
    config: RunConfig,
    *,
    run_root: str | Path,
    sampled_continuation_seed: int,
    resume: bool = False,
    loader: Callable[[RunConfig], BackboneHandle] = load_backbone,
) -> CampaignResult:
    """Run the final A40 calibration in one process and one model load.

    The ``loader`` seam exists solely for weight-free scheduler tests.  The
    production command always uses :func:`load_backbone`.
    """
    config.validate()
    if sampled_continuation_seed not in config.identity.continuation_seeds:
        raise CampaignError("sampled continuation seed is not in the frozen config")
    if not config.identity_mode():
        raise CampaignError("SPEC-02 requires all Formic flags and boundary hooks OFF")
    _assert_a40_environment()
    if git_dirty() is True:
        raise CampaignError("refusing a GPU campaign from a dirty worktree")
    commit = git_commit()
    if not commit:
        raise CampaignError("unable to resolve the campaign git commit")
    root = Path(run_root)
    if root.exists() and any(root.iterdir()) and not resume:
        raise CampaignError("run directory already contains artifacts; pass --resume")
    if resume and not (root / "manifest.json").is_file():
        raise CampaignError("--resume requires an existing campaign manifest")
    corpus = load_frozen_corpus(_repo_path(config.identity.prompt_set_path))
    plan = build_campaign_plan(config, corpus)
    started_at = _now()
    memory = IncrementalMemoryWriter(root / "memory" / "cuda_memory.json")
    memory.record("before_load")

    # Exactly one invocation of the production loader occurs in this function.
    handle = loader(config)
    try:
        memory.record("after_load", handle.model)
        corpus.validate_tokenizer(handle.tokenizer)
        backbone = canonical_backbone_hash(handle.inventory)
        expected_backbone = load_reusable_backbone_hash(
            _repo_path(config.identity.backbone_hash_path)
        )
        if backbone != expected_backbone:
            raise CampaignError(
                "loaded backbone differs from the committed audited backbone hash"
            )
        identity = CampaignIdentity(
            protocol="SPEC-02-h8-option-b-balanced-v2",
            config_sha256=config.config_hash(),
            corpus_sha256=corpus.corpus_sha256,
            git_commit=commit,
            backbone_sha256=backbone.sha256,
        )
        writer = IncrementalCampaignWriter(root, identity)
        writer.validate()
        _write_backbone_hash(root, backbone)
        atomic_write_json(
            root / "run_metadata.json",
            {
                "schema_version": 1,
                "started_at": started_at,
                "resume": resume,
                "identity": identity.__dict__,
                "environment": environment_report(),
                "backbone": backbone.to_dict(),
                "sampled_continuation_seed": sampled_continuation_seed,
            },
        )
        session = _MeasurementSession(handle, config, memory=memory)
        session.plan = plan
        _phase_preflight(writer, root, handle, plan, memory=memory)
        release_cuda_working_set()
        _phase_trace_inertness(writer, session, corpus)
        release_cuda_working_set()
        _phase_legacy(writer, session, corpus)
        release_cuda_working_set()
        noise_floor = _phase_noise_floor(writer, session, corpus)
        release_cuda_working_set()
        snapshot_evidence = _phase_snapshot_restore(writer, session, corpus)
        release_cuda_working_set()
        continuations = _phase_continuations(writer, session, corpus)
        release_cuda_working_set()
        calibration = _phase_calibration(
            writer,
            session,
            plan,
            continuations,
            sampled_continuation_seed,
        )
        release_cuda_working_set()
        _phase_probe64(writer, session, corpus, continuations, sampled_continuation_seed)
        release_cuda_working_set()
        raw_path = root / "calibration" / "raw_measurements.json"
        atomic_write_json(raw_path, {"schema_version": 1, "observations": calibration})
        raw_digest = sha256_bytes(raw_path.read_bytes())
        candidate = build_candidate_tolerances(
            calibration,
            raw_measurements_sha256=raw_digest,
            reference_floor_observations=_flatten_reference_floor(noise_floor),
        )
        candidate_path = root / "tolerances.candidate.json"
        atomic_write_json(candidate_path, candidate)
        snapshot_adjudication = _adjudicate_snapshot_candidate(
            snapshot_evidence, candidate
        )
        atomic_write_json(
            root / "snapshot_restore" / "adjudication.candidate.json",
            snapshot_adjudication,
        )
        if snapshot_adjudication["verdict"] != "CANDIDATE_PASS":
            raise CampaignError("snapshot/restore candidate adjudication failed")
        verdict = candidate_verdict(calibration)
        verdict.update(
            {
                "schema_version": 1,
                "spec": "SPEC-02",
                "status": "candidate_review_required",
                "generated_at": _now(),
                "config_sha256": config.config_hash(),
                "corpus_sha256": corpus.corpus_sha256,
                "backbone_sha256": backbone.sha256,
                "raw_measurements_sha256": raw_digest,
                "tolerances_candidate_sha256": sha256_bytes(candidate_path.read_bytes()),
            }
        )
        atomic_write_json(root / "verdict.candidate.json", verdict)
        atomic_write_json(
            root / "terminal.json",
            {
                "schema_version": 1,
                "status": "CALIBRATION_COMPLETE",
                "message": "CALIBRATION COMPLETE — PROMOTION REQUIRED",
                "finished_at": _now(),
                "pod_action_required": None,
            },
        )
        return CampaignResult(
            True,
            root,
            "CALIBRATION COMPLETE — PROMOTION REQUIRED",
            verdict,
        )
    except Exception as exc:
        # Preserve the allocator state at the point of failure.  This is a
        # diagnostic observation only; it does not keep the run alive or
        # alter the identity protocol.
        try:
            memory.record("on_failure", handle.model)
            memory.write_live_summary(handle.model)
        except Exception:
            pass
        _write_failure(root, exc)
        raise
    finally:
        # The caller exits immediately after this function.  No post-run
        # analysis is performed in the GPU process.
        del handle
        gc.collect()


def _phase_preflight(
    writer: IncrementalCampaignWriter,
    root: Path,
    handle: BackboneHandle,
    plan: CampaignPlan,
    *,
    memory: IncrementalMemoryWriter | None = None,
) -> None:
    if "preflight" in writer.completed_phases():
        estimate = report_estimate(
            load_preflight_estimate(root / "preflight" / "estimate.json")
        )
        _report_preflight_estimate(root, estimate.to_dict())
        return
    run = run_preflight(
        handle,
        plan,
        estimate_path=root / "preflight" / "estimate.json",
        details_path=root / "preflight" / "timings.json",
        memory_observer=(
            (lambda label: memory.record(label, handle.model))
            if memory is not None
            else None
        ),
    )
    estimate = report_estimate(run.estimate)
    _report_preflight_estimate(root, estimate.to_dict())
    writer.write_phase(
        "preflight",
        {
            "schema_version": 1,
            "forwards": run.estimate.preflight_forwards,
            "estimate": estimate.to_dict(),
        },
    )


def _report_preflight_estimate(root: Path, expected: dict[str, Any]) -> None:
    """Invoke the non-blocking reporter and verify its informational output."""
    canonical_preflight = _repo_path("artifacts/step2/preflight")
    atomic_write_json(
        canonical_preflight / "estimate.json",
        json.loads((root / "preflight" / "estimate.json").read_text(encoding="utf-8")),
    )
    reporter = subprocess.run(
        (
            sys.executable,
            str(_repo_path("scripts/step2_budget_gate.py")),
            "--preflight",
            str(root / "preflight" / "estimate.json"),
            "--output",
            str(canonical_preflight / "estimate_report.json"),
        ),
        cwd=_repo_path("."),
        check=True,
    )
    if reporter.returncode != 0:  # pragma: no cover - check=True is defensive
        raise CampaignError("post-preflight estimate reporter failed")
    reported = json.loads(
        (canonical_preflight / "estimate_report.json").read_text(encoding="utf-8")
    )
    if reported != expected:
        raise CampaignError("post-preflight estimate reporter changed the estimate")
    atomic_write_json(root / "preflight" / "estimate_report.json", reported)


def _phase_trace_inertness(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
) -> None:
    phase = "trace_inertness"
    if phase in writer.completed_phases():
        return
    records: list[dict[str, Any]] = []
    for prompt in corpus.prompts:
        case_id = f"trace__{prompt.id}"
        if case_id in writer.completed_cases():
            records.append(_read_case(writer, case_id))
            continue
        for _ in range(session.config.numerics.warmup_traces_per_shape):
            session.trace_off_prefill(prompt)
        pairs = []
        for repetition in range(session.config.identity.exact_gate_repetitions):
            if not records and repetition == 0 and session.memory is not None:
                release_cuda_working_set()
                session.memory.record("before_first_comparison", session.handle.model)
            metric = session.trace_off_on_pair(prompt)
            if not metric.tensor.exact or not metric.top1_agreement:
                raise CampaignError(
                    f"trace inertness failed for {prompt.id}: delta={metric.tensor.max_abs_delta} "
                    f"top1={metric.top1_agreement}"
                )
            pairs.append({"repetition": repetition, "metric": metric.to_dict()})
        payload = {
            "schema_version": 1,
            "phase": phase,
            "prompt_id": prompt.id,
            "warmup_paths": session.config.numerics.warmup_traces_per_shape,
            "pairs": pairs,
        }
        writer.write_case(case_id, payload)
        records.append(payload)
    writer.write_phase(phase, {"schema_version": 1, "forwards": 120, "cases": records})


def _phase_legacy(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
) -> None:
    phase = "legacy_continuity"
    if phase in writer.completed_phases():
        return
    records: list[dict[str, Any]] = []
    for prompt in (item for item in corpus.prompts if item.set_name == "legacy"):
        case_id = f"legacy__{prompt.id}"
        payload = _measure_balanced_or_resume(
            writer,
            case_id,
            phase,
            session,
            prompt,
            forced_token_ids=timing_continuation(prompt, session.config.identity.decode_tokens),
            repetitions=session.config.identity.exact_gate_repetitions,
            exact_required=True,
        )
        records.append(payload)
    writer.write_phase(phase, {"schema_version": 1, "forwards": 3_552, "cases": records})


def _phase_noise_floor(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
) -> list[dict[str, Any]]:
    phase = "noise_floor"
    if phase in writer.completed_phases():
        phase_payload = json.loads(
            (writer.phases_dir / f"{phase}.json").read_text(encoding="utf-8")
        )
        return list(phase_payload["cases"])
    by_id = {item.id: item for item in corpus.prompts}
    records: list[dict[str, Any]] = []
    for prompt_id in ("audit_echo", "short_error_assertion", "medium_cache_regression"):
        prompt = by_id[prompt_id]
        case_id = f"noise__{prompt.id}__alternating"
        records.append(
            _measure_noise_floor_or_resume(
                writer,
                case_id,
                phase,
                session,
                prompt,
                forced_token_ids=timing_continuation(
                    prompt, session.config.identity.decode_tokens
                ),
                repetitions=session.config.identity.measurement_repetitions,
            )
        )
    writer.write_phase(phase, {"schema_version": 1, "forwards": 624, "cases": records})
    return records


def _phase_snapshot_restore(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
) -> dict[str, Any]:
    phase = "snapshot_restore"
    if phase in writer.completed_phases():
        phase_payload = json.loads(
            (writer.phases_dir / f"{phase}.json").read_text(encoding="utf-8")
        )
        return dict(phase_payload["cases"][0])
    prompt = next(item for item in corpus.prompts if item.id == session.config.identity.snapshot_validation_prompt_id)
    case_id = "snapshot_restore__audit_echo"
    if case_id in writer.completed_cases():
        payload = _read_case(writer, case_id)
    else:
        payload = session.snapshot_restore(prompt)
        writer.write_case(case_id, payload)
    writer.write_phase(phase, {"schema_version": 1, "forwards": 48, "cases": [payload]})
    return payload


def _phase_continuations(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
) -> dict[tuple[str, int | None], tuple[int, ...]]:
    phase = "reference_continuations"
    by_id = {item.id: item for item in corpus.prompts}
    result: dict[tuple[str, int | None], tuple[int, ...]] = {}
    if phase in writer.completed_phases():
        phase_payload = json.loads((writer.phases_dir / f"{phase}.json").read_text(encoding="utf-8"))
        for item in phase_payload["cases"]:
            result[(item["prompt_id"], item["seed"])] = tuple(item["token_ids"])
        return result
    records: list[dict[str, Any]] = []
    for prompt_id in session.config.identity.decode_prompt_ids:
        prompt = by_id[prompt_id]
        greedy = session.greedy_continuation(prompt)
        greedy_payload = {
            "schema_version": 1,
            "phase": phase,
            "prompt_id": prompt.id,
            "sampling": SamplingMode.GREEDY.value,
            "seed": None,
            "token_ids": list(greedy),
        }
        writer.write_case(f"continuation__{prompt.id}__greedy", greedy_payload)
        records.append(greedy_payload)
        result[(prompt.id, None)] = greedy
        for seed in session.config.identity.continuation_seeds:
            forced = session.sampled_continuation(prompt, seed)
            payload = {
                "schema_version": 1,
                "phase": phase,
                "prompt_id": prompt.id,
                "sampling": SamplingMode.SEEDED_SAMPLING.value,
                "seed": seed,
                "token_ids": list(forced),
            }
            writer.write_case(f"continuation__{prompt.id}__s{seed}", payload)
            records.append(payload)
            result[(prompt.id, seed)] = forced
    writer.write_phase(phase, {"schema_version": 1, "forwards": 96, "cases": records})
    return result


def _phase_calibration(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    plan: CampaignPlan,
    continuations: dict[tuple[str, int | None], tuple[int, ...]],
    sampled_seed: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for length_class in ("short", "medium", "long"):
        if length_class in writer.completed_phases():
            phase_payload = json.loads((writer.phases_dir / f"{length_class}.json").read_text(encoding="utf-8"))
            observations.extend(_flatten_observations(phase_payload["cases"]))
            continue
        records: list[dict[str, Any]] = []
        for path in (item for item in plan.calibration_paths if item.prompt.length_class == length_class):
            if path.mode is ExecutionMode.DECODE_CACHED:
                greedy = _measure_or_resume(
                    writer,
                    f"calibration__{path.key}__greedy",
                    length_class,
                    session,
                    path,
                    forced_token_ids=continuations[(path.prompt.id, None)],
                    repetitions=session.config.identity.measurement_repetitions,
                    sampling=SamplingMode.GREEDY,
                    continuation_seed=None,
                    exact_required=False,
                )
                sampled = _measure_or_resume(
                    writer,
                    f"calibration__{path.key}__sampled_s{sampled_seed}",
                    length_class,
                    session,
                    path,
                    forced_token_ids=continuations[(path.prompt.id, sampled_seed)],
                    repetitions=session.config.identity.measurement_repetitions,
                    sampling=SamplingMode.SEEDED_SAMPLING,
                    continuation_seed=sampled_seed,
                    exact_required=False,
                )
                records.extend((greedy, sampled))
            elif path.mode is ExecutionMode.DECODE_RECOMPUTE:
                for sampling, forced, seed in (
                    (
                        SamplingMode.GREEDY,
                        continuations[(path.prompt.id, None)],
                        None,
                    ),
                    (SamplingMode.SEEDED_SAMPLING, continuations[(path.prompt.id, sampled_seed)], sampled_seed),
                ):
                    records.append(
                        _measure_or_resume(
                            writer,
                            f"calibration__{path.key}__{sampling.value}__s{seed}",
                            length_class,
                            session,
                            path,
                            forced_token_ids=forced,
                            repetitions=session.config.identity.measurement_repetitions,
                            sampling=sampling,
                            continuation_seed=seed,
                            exact_required=False,
                        )
                    )
            else:
                records.append(
                    _measure_or_resume(
                        writer,
                        f"calibration__{path.key}",
                        length_class,
                        session,
                        path,
                        forced_token_ids=(),
                        repetitions=session.config.identity.measurement_repetitions,
                        sampling=SamplingMode.GREEDY,
                        continuation_seed=None,
                        exact_required=False,
                    )
                )
        writer.write_phase(
            length_class,
            {
                "schema_version": 1,
                "forwards": session.plan.phase_forwards[length_class] if session.plan else None,
                "cases": records,
            },
        )
        observations.extend(_flatten_observations(records))
    return observations


def _phase_probe64(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    corpus: FrozenPromptCorpus,
    continuations: dict[tuple[str, int | None], tuple[int, ...]],
    sampled_seed: int,
) -> None:
    phase = "accumulation_probe_64"
    if phase in writer.completed_phases():
        return
    by_id = {item.id: item for item in corpus.prompts}
    records: list[dict[str, Any]] = []
    for prompt_id in session.config.identity.accumulation_probe_prompt_ids:
        prompt = by_id[prompt_id]
        path = CampaignPath(prompt, ExecutionMode.DECODE_CACHED)
        # The 64-token probe is logits-only and uses a repeated frozen token to
        # avoid generating 64 additional stochastic reference forwards.
        forced = timing_continuation(prompt, session.config.identity.accumulation_probe_tokens)
        records.append(
            _measure_or_resume(
                writer,
                f"probe64__{prompt.id}",
                phase,
                session,
                path,
                forced_token_ids=forced,
                repetitions=session.config.identity.exact_gate_repetitions,
                sampling=SamplingMode.GREEDY,
                continuation_seed=None,
                exact_required=False,
                logits_only=True,
                decode_steps=session.config.identity.accumulation_probe_tokens,
            )
        )
    writer.write_phase(phase, {"schema_version": 1, "forwards": 2_048, "cases": records})


def _measure_balanced_or_resume(
    writer: IncrementalCampaignWriter,
    case_id: str,
    phase: str,
    session: "_MeasurementSession",
    prompt: FrozenPrompt,
    *,
    forced_token_ids: tuple[int, ...],
    repetitions: int,
    exact_required: bool,
) -> dict[str, Any]:
    if case_id in writer.completed_cases():
        return _read_case(writer, case_id)
    latest: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "phase": phase,
        "case_id": case_id,
        "prompt_id": prompt.id,
    }

    def observe(payload: dict[str, Any]) -> None:
        nonlocal latest
        latest = {
            **payload,
            "phase": phase,
            "case_id": case_id,
            "prompt_id": prompt.id,
        }
        writer.write_diagnostic(case_id, latest)

    try:
        payload = session.measure_balanced_logits(
            prompt,
            forced_token_ids=forced_token_ids,
            repetitions=repetitions,
            observer=observe,
        )
        payload.update(
            {
                "phase": phase,
                "case_id": case_id,
                "prompt_id": prompt.id,
                "length_class": prompt.length_class,
                "exact_prompt_length": len(prompt.token_ids),
            }
        )
        if exact_required and not payload["matched_endpoint_exact"]:
            raise CampaignError(f"balanced exact gate diverged: {case_id}")
    except Exception as exc:
        writer.write_diagnostic(
            case_id,
            {
                **latest,
                "status": "FAILED",
                "failure": {"exception": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    writer.write_case(case_id, payload)
    return payload


def _measure_noise_floor_or_resume(
    writer: IncrementalCampaignWriter,
    case_id: str,
    phase: str,
    session: "_MeasurementSession",
    prompt: FrozenPrompt,
    *,
    forced_token_ids: tuple[int, ...],
    repetitions: int,
) -> dict[str, Any]:
    if case_id in writer.completed_cases():
        return _read_case(writer, case_id)
    latest: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "phase": phase,
        "case_id": case_id,
        "prompt_id": prompt.id,
    }

    def observe(payload: dict[str, Any]) -> None:
        nonlocal latest
        latest = {
            **payload,
            "phase": phase,
            "case_id": case_id,
            "prompt_id": prompt.id,
        }
        writer.write_diagnostic(
            case_id,
            latest,
        )

    try:
        payload = session.measure_noise_floor_logits(
            prompt,
            forced_token_ids=forced_token_ids,
            repetitions=repetitions,
            observer=observe,
        )
        payload.update(
            {
                "phase": phase,
                "case_id": case_id,
                "prompt_id": prompt.id,
                "length_class": prompt.length_class,
                "exact_prompt_length": len(prompt.token_ids),
            }
        )
    except Exception as exc:
        writer.write_diagnostic(
            case_id,
            {
                **latest,
                "status": "FAILED",
                "failure": {"exception": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    writer.write_case(case_id, payload)
    return payload


def _measure_or_resume(
    writer: IncrementalCampaignWriter,
    case_id: str,
    phase: str,
    session: "_MeasurementSession",
    path: CampaignPath,
    *,
    forced_token_ids: tuple[int, ...],
    repetitions: int,
    sampling: SamplingMode,
    continuation_seed: int | None,
    exact_required: bool,
    endpoints: tuple[Endpoint, Endpoint] | None = None,
    logits_only: bool = False,
    decode_steps: int | None = None,
) -> dict[str, Any]:
    if case_id in writer.completed_cases():
        return _read_case(writer, case_id)
    latest: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "phase": phase,
        "case_id": case_id,
    }

    def observe(payload: dict[str, Any]) -> None:
        nonlocal latest
        latest = payload
        writer.write_diagnostic(case_id, latest)

    try:
        payload = session.measure_forced(
            case_id=case_id,
            phase=phase,
            path=path,
            forced_token_ids=forced_token_ids,
            repetitions=repetitions,
            sampling=sampling,
            continuation_seed=continuation_seed,
            exact_required=exact_required,
            endpoints=endpoints,
            logits_only=logits_only,
            decode_steps=decode_steps,
            repetition_observer=observe,
        )
    except Exception as exc:
        writer.write_diagnostic(
            case_id,
            {
                **latest,
                "status": "FAILED",
                "failure": {"exception": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    writer.write_case(case_id, payload)
    return payload


def _measure_greedy_or_resume(
    writer: IncrementalCampaignWriter,
    session: "_MeasurementSession",
    path: CampaignPath,
) -> dict[str, Any]:
    case_id = f"calibration__{path.key}__greedy"
    if case_id in writer.completed_cases():
        return _read_case(writer, case_id)
    latest: dict[str, Any] = {
        "schema_version": 1,
        "status": "STARTED",
        "phase": path.prompt.length_class,
        "case_id": case_id,
    }

    def observe(payload: dict[str, Any]) -> None:
        nonlocal latest
        latest = payload
        writer.write_diagnostic(case_id, latest)

    try:
        payload = session.measure_greedy(case_id, path, repetition_observer=observe)
    except Exception as exc:
        writer.write_diagnostic(
            case_id,
            {
                **latest,
                "status": "FAILED",
                "failure": {"exception": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    writer.write_case(case_id, payload)
    return payload


class _MeasurementSession:
    def __init__(
        self,
        handle: BackboneHandle,
        config: RunConfig,
        *,
        memory: IncrementalMemoryWriter | None = None,
    ) -> None:
        self.handle = handle
        self.config = config
        self.reference = Endpoint("reference", handle.model, handle.view, False)
        self.runner = Endpoint("runner", handle.model, handle.view, True)
        self.warmups = SharedShapeWarmups(config.numerics.warmup_traces_per_shape)
        self.balanced_warmups = SharedShapeWarmups(
            config.numerics.warmup_traces_per_shape
        )
        self.plan: CampaignPlan | None = None
        self.memory = memory

    def warm(self, path: CampaignPath, forced: tuple[int, ...], decode_steps: int) -> int:
        candidate_shapes = expected_shapes(
            prompt_length=len(path.prompt.token_ids),
            mode=path.mode,
            segmentation=path.segmentation,
            decode_steps=decode_steps,
        )
        reference_shapes = reference_shapes_for_candidate(
            prompt_length=len(path.prompt.token_ids),
            candidate_mode=path.mode,
            segmentation=path.segmentation,
            decode_steps=decode_steps,
            length_class=path.prompt.length_class,
        )
        reference_count = self.warmups.required_path_traces(reference_shapes)
        for _ in range(reference_count):
            execute_reference_for_candidate(
                self.reference,
                prompt_token_ids=path.prompt.token_ids,
                candidate_mode=path.mode,
                length_class=path.prompt.length_class,
                segmentation=path.segmentation,
                forced_token_ids=forced,
                capture=False,
            )
            self.warmups.record_path_trace(reference_shapes)
        candidate_count = self.warmups.required_path_traces(candidate_shapes)
        for _ in range(candidate_count):
            execute_path(
                self.runner,
                prompt_token_ids=path.prompt.token_ids,
                mode=path.mode,
                length_class=path.prompt.length_class,
                segmentation=path.segmentation,
                forced_token_ids=forced,
                capture=False,
            )
            self.warmups.record_path_trace(candidate_shapes)
        return reference_count + candidate_count

    def measure_balanced_logits(
        self,
        prompt: FrozenPrompt,
        *,
        forced_token_ids: tuple[int, ...],
        repetitions: int,
        observer: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        shapes = expected_shapes(
            prompt_length=len(prompt.token_ids),
            mode=ExecutionMode.DECODE_CACHED,
            segmentation=None,
            decode_steps=len(forced_token_ids),
        )
        warmups = self.balanced_warmups.required_path_traces(shapes)
        payload = run_balanced_logits_gate(
            self.reference,
            self.runner,
            prompt_token_ids=prompt.token_ids,
            forced_token_ids=forced_token_ids,
            repetitions=repetitions,
            warmup_pair_traces=warmups,
            observer=observer,
        )
        for _ in range(warmups):
            self.balanced_warmups.record_path_trace(shapes)
        return payload

    def measure_noise_floor_logits(
        self,
        prompt: FrozenPrompt,
        *,
        forced_token_ids: tuple[int, ...],
        repetitions: int,
        observer: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        shapes = expected_shapes(
            prompt_length=len(prompt.token_ids),
            mode=ExecutionMode.DECODE_CACHED,
            segmentation=None,
            decode_steps=len(forced_token_ids),
        )
        warmups = self.balanced_warmups.required_path_traces(shapes)
        payload = run_alternating_noise_floor(
            self.reference,
            self.runner,
            prompt_token_ids=prompt.token_ids,
            forced_token_ids=forced_token_ids,
            repetitions=repetitions,
            warmup_pair_traces=warmups,
            observer=observer,
        )
        for _ in range(warmups):
            self.balanced_warmups.record_path_trace(shapes)
        return payload

    def measure_forced(
        self,
        *,
        case_id: str,
        phase: str,
        path: CampaignPath,
        forced_token_ids: tuple[int, ...],
        repetitions: int,
        sampling: SamplingMode,
        continuation_seed: int | None,
        exact_required: bool,
        endpoints: tuple[Endpoint, Endpoint] | None,
        logits_only: bool,
        decode_steps: int | None,
        repetition_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        steps = decode_steps or self.config.identity.decode_tokens
        warm_forced = forced_token_ids or timing_continuation(path.prompt, steps)
        warmups = self.warm(path, warm_forced, steps)
        reference, candidate = endpoints or (self.reference, self.runner)
        results: list[dict[str, Any]] = []
        fingerprints: list[tuple[str, str]] = []
        for repetition in range(repetitions):
            if endpoints is None:
                pair = run_cross_path_pair(
                    reference,
                    candidate,
                    prompt_token_ids=path.prompt.token_ids,
                    candidate_mode=path.mode,
                    length_class=path.prompt.length_class,
                    segmentation=path.segmentation,
                    forced_token_ids=forced_token_ids,
                    capture=True,
                )
            else:
                pair = run_aligned_pair(
                    reference,
                    candidate,
                    prompt_token_ids=path.prompt.token_ids,
                    mode=path.mode,
                    length_class=path.prompt.length_class,
                    segmentation=path.segmentation,
                    forced_token_ids=forced_token_ids,
                    capture=True,
                )
            fingerprints.append((pair.reference_fingerprint, pair.candidate_fingerprint))
            results.append(
                _pair_to_observation(
                    pair,
                    case_id=case_id,
                    prompt=path.prompt,
                    mode=path.mode,
                    segmentation=path.segmentation,
                    sampling=sampling,
                    continuation_seed=continuation_seed,
                    repetition=repetition,
                    logits_only=logits_only,
                )
            )
            if repetition_observer is not None:
                repetition_observer(
                    _measurement_payload(
                        status="MEASURING",
                        phase=phase,
                        case_id=case_id,
                        prompt_id=path.prompt.id,
                        warmups=warmups,
                        results=results,
                        fingerprints=fingerprints,
                    )
                )
            del pair
        _assert_stable(fingerprints, case_id)
        if exact_required:
            _assert_exact(results, case_id)
        payload = _measurement_payload(
            status="COMPLETE",
            phase=phase,
            case_id=case_id,
            prompt_id=path.prompt.id,
            warmups=warmups,
            results=results,
            fingerprints=fingerprints,
        )
        if repetition_observer is not None:
            repetition_observer(payload)
        return payload

    def measure_greedy(
        self,
        case_id: str,
        path: CampaignPath,
        *,
        repetition_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        steps = self.config.identity.decode_tokens
        warmups = self.warm(path, timing_continuation(path.prompt, steps), steps)
        results: list[dict[str, Any]] = []
        fingerprints: list[tuple[str, str]] = []
        for repetition in range(self.config.identity.measurement_repetitions):
            pair = run_greedy_pair(
                self.reference,
                self.runner,
                prompt_token_ids=path.prompt.token_ids,
                length_class=path.prompt.length_class,
                decode_steps=steps,
                capture=True,
            )
            fingerprints.append((pair.reference_fingerprint, pair.candidate_fingerprint))
            results.append(
                _pair_to_observation(
                    pair,
                    case_id=case_id,
                    prompt=path.prompt,
                    mode=ExecutionMode.DECODE_CACHED,
                    segmentation=None,
                    sampling=SamplingMode.GREEDY,
                    continuation_seed=None,
                    repetition=repetition,
                    logits_only=False,
                )
            )
            if repetition_observer is not None:
                repetition_observer(
                    _measurement_payload(
                        status="MEASURING",
                        phase=path.prompt.length_class,
                        case_id=case_id,
                        prompt_id=path.prompt.id,
                        warmups=warmups,
                        results=results,
                        fingerprints=fingerprints,
                    )
                )
            del pair
        _assert_stable(fingerprints, case_id)
        payload = _measurement_payload(
            status="COMPLETE",
            phase=path.prompt.length_class,
            case_id=case_id,
            prompt_id=path.prompt.id,
            warmups=warmups,
            results=results,
            fingerprints=fingerprints,
        )
        if repetition_observer is not None:
            repetition_observer(payload)
        return payload

    @torch.no_grad()
    def trace_off_prefill(self, prompt: FrozenPrompt) -> None:
        device = self.handle.device
        token_ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
        self.handle.model(input_ids=token_ids, use_cache=False)

    @torch.no_grad()
    def trace_off_on_pair(self, prompt: FrozenPrompt):
        device = self.handle.device
        token_ids = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
        off = self.handle.model(input_ids=token_ids, use_cache=False).logits[0, -1].detach().clone()
        profile = (
            CaptureProfile.FINAL_STATE_ONLY
            if prompt.length_class == "long"
            else CaptureProfile.FULL_BOUNDARIES
        )
        collector = IdentityTraceCollector(
            model=self.handle.model,
            view=self.handle.view,
            cache=None,
            capture_profile=profile,
        )
        from formic.backbone.runner import identity_forward

        identity_forward(
            SimpleNamespace(model=self.handle.model),
            trace_collector=collector,
            input_ids=token_ids,
            use_cache=False,
        )
        assert collector.last_trace is not None
        return compare_logits(off, collector.last_trace.logits)

    @torch.no_grad()
    def greedy_continuation(self, prompt: FrozenPrompt) -> tuple[int, ...]:
        return self._reference_continuation(prompt, seed=0, mode="greedy")

    @torch.no_grad()
    def sampled_continuation(self, prompt: FrozenPrompt, seed: int) -> tuple[int, ...]:
        return self._reference_continuation(
            prompt, seed=seed, mode="seeded_sampling"
        )

    def _reference_continuation(
        self, prompt: FrozenPrompt, *, seed: int, mode: str
    ) -> tuple[int, ...]:
        from transformers.cache_utils import DynamicCache

        device = self.handle.device
        cache = DynamicCache(config=self.handle.model.config)
        calls = 0

        def next_logits(context: tuple[int, ...]) -> torch.Tensor:
            nonlocal calls
            token_ids = context if calls == 0 else context[-1:]
            inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
            calls += 1
            return self.handle.model(
                input_ids=inputs,
                past_key_values=cache,
                use_cache=True,
            ).logits[0, -1]

        result = generate_forced_continuation(
            prompt_token_ids=prompt.token_ids,
            steps=self.config.identity.decode_tokens,
            seed=seed,
            mode=mode,
            reference_next_logits=next_logits,
            temperature=self.config.sampling.payload.temperature,
            top_p=self.config.sampling.payload.top_p,
            top_k=self.config.sampling.payload.top_k,
        )
        if calls != self.config.identity.decode_tokens:
            raise CampaignError("reference continuation forward count changed")
        return result.token_ids

    @torch.no_grad()
    def snapshot_restore(self, prompt: FrozenPrompt) -> dict[str, Any]:
        from transformers.cache_utils import DynamicCache

        steps = self.config.identity.decode_tokens
        forced = timing_continuation(prompt, steps)
        device = self.handle.device
        observations: list[dict[str, Any]] = []
        for repetition in range(3):
            cache = DynamicCache(config=self.handle.model.config)
            current = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
            uninterrupted: list[torch.Tensor] = []
            frozen = None
            for step in range(steps):
                output = self.handle.model(input_ids=current, past_key_values=cache, use_cache=True)
                uninterrupted.append(output.logits[0, -1].detach().clone())
                if step == 3:
                    frozen = snapshot(
                        model=self.handle.model,
                        cache=cache,
                        position=PositionState(sequence_length=int(cache.get_seq_length())),
                    )
                if step < steps - 1:
                    current = torch.tensor([[forced[step]]], dtype=torch.long, device=device)
            if frozen is None:
                raise CampaignError("snapshot midpoint was not reached")
            controller = ExecutionStateController(self.handle.model)
            branch_a = controller.restore(frozen)
            branch_b = controller.restore(frozen)
            source_storage = {tensor_storage_identity(item) for _, item in iter_snapshot_tensors(frozen)}
            a_storage = _branch_storage(branch_a)
            b_storage = _branch_storage(branch_b)
            if source_storage & a_storage or source_storage & b_storage or a_storage & b_storage:
                raise CampaignError("snapshot fork storage isolation failed")
            frozen_before = snapshot_fingerprint(frozen)
            b_before = _live_cache_fingerprint(branch_b.cache)
            # Snapshot is taken after step 3: cache contains prompt plus the
            # first three forced IDs, so steps 4..7 consume IDs 3..6.
            tail_tokens = forced[3:7]
            a_logits = _continue_branch(controller, branch_a, tail_tokens, device)
            if _live_cache_fingerprint(branch_b.cache) != b_before or snapshot_fingerprint(frozen) != frozen_before:
                raise CampaignError("snapshot fork mutation leaked from branch A")
            b_logits = _continue_branch(controller, branch_b, tail_tokens, device)
            comparisons = [compare_logits(uninterrupted[4 + index], value) for index, value in enumerate(a_logits)]
            comparisons += [compare_logits(uninterrupted[4 + index], value) for index, value in enumerate(b_logits)]
            serialised = [item.to_dict() for item in comparisons]
            comparison_fingerprint = sha256_bytes(
                json.dumps(
                    serialised, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            observations.append(
                {
                    "repetition": repetition,
                    "all_exact": all(item.tensor.exact and item.top1_agreement for item in comparisons),
                    "max_abs_delta": max(item.tensor.max_abs_delta for item in comparisons),
                    "comparison_fingerprint": comparison_fingerprint,
                    "comparisons": serialised,
                }
            )
        last_two_exact = (
            observations[-2]["comparison_fingerprint"]
            == observations[-1]["comparison_fingerprint"]
        )
        if not last_two_exact:
            raise InvalidMeasurement(
                "snapshot/restore last two comparison traces are unstable"
            )
        return {
            "schema_version": 1,
            "phase": "snapshot_restore",
            "case_id": "snapshot_restore__audit_echo",
            "prompt_id": prompt.id,
            "observations": observations,
            "stability": {
                "assertion": "last_two_snapshot_contrasts_exact",
                "last_two_exact": last_two_exact,
            },
        }


def _measurement_payload(
    *,
    status: str,
    phase: str,
    case_id: str,
    prompt_id: str,
    warmups: int,
    results: list[dict[str, Any]],
    fingerprints: list[tuple[str, str]],
) -> dict[str, Any]:
    """Serialise measured repetitions plus their cross-repetition evidence."""
    return {
        "schema_version": 1,
        "status": status,
        "phase": phase,
        "case_id": case_id,
        "prompt_id": prompt_id,
        "warmup_paths": warmups,
        "repetitions": list(results),
        "stability": _stability_details(fingerprints),
    }


def _stability_details(fingerprints: list[tuple[str, str]]) -> dict[str, Any]:
    """Return diagnostic evidence for the pinned last-two-traces assertion."""
    entries = [
        {
            "repetition": index,
            "reference_fingerprint": reference,
            "candidate_fingerprint": candidate,
        }
        for index, (reference, candidate) in enumerate(fingerprints)
    ]
    last_two_exact = (
        None
        if len(fingerprints) < 2
        else fingerprints[-2] == fingerprints[-1]
    )
    first_changed = next(
        (
            index
            for index in range(1, len(fingerprints))
            if fingerprints[index] != fingerprints[index - 1]
        ),
        None,
    )
    return {
        "assertion": "last_two_pairs_exact",
        "last_two_exact": last_two_exact,
        "first_changed_repetition": first_changed,
        "fingerprints": entries,
    }


def _pair_to_observation(
    pair: Any,
    *,
    case_id: str,
    prompt: FrozenPrompt,
    mode: ExecutionMode,
    segmentation: str | None,
    sampling: SamplingMode,
    continuation_seed: int | None,
    repetition: int,
    logits_only: bool,
) -> dict[str, Any]:
    payload = pair.payload
    if not isinstance(payload, AlignedCasePayload):
        raise CampaignError("measured pair omitted its trace payload")
    measurements: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    for comparison in payload.comparisons:
        measurements.extend(_serialise_comparison(comparison, logits_only=logits_only))
        if not logits_only:
            not_applicable.extend(
                {
                    "step": comparison.measurements[0].step,
                    "location": {
                        "point": item.point.value,
                        "boundary": item.boundary,
                        "layer": item.layer,
                        "component": item.component,
                    },
                }
                for item in comparison.not_applicable
            )
    return {
        "case_id": case_id,
        "prompt_id": prompt.id,
        "length_class": prompt.length_class,
        "exact_prompt_length": len(prompt.token_ids),
        "mode": mode.value,
        "segmentation": segmentation,
        "sampling": sampling.value,
        "continuation_seed": continuation_seed,
        "repetition": repetition,
        "comparison_protocol": "canonical_reference_vs_formic_candidate",
        "reference_input_shapes": [frame.shape.key for frame in payload.reference.frames],
        "candidate_input_shapes": [frame.shape.key for frame in payload.candidate.frames],
        "reference_fingerprint": pair.reference_fingerprint,
        "candidate_fingerprint": pair.candidate_fingerprint,
        "captured_state_tensors": pair.captured_state_tensors,
        "measurements": measurements,
        "not_applicable": not_applicable,
    }


def _serialise_comparison(comparison: TraceComparison, *, logits_only: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in comparison.measurements:
        if logits_only and item.location.point.value != "logits":
            continue
        metric = item.metric.to_dict()
        result.append(
            {
                "step": item.step,
                "location": {
                    "point": item.location.point.value,
                    "boundary": item.location.boundary,
                    "layer": item.location.layer,
                    "component": item.location.component,
                },
                "metric": metric,
            }
        )
    return result


def _assert_stable(fingerprints: list[tuple[str, str]], case_id: str) -> None:
    if len(fingerprints) < 2 or fingerprints[-1] != fingerprints[-2]:
        details = _stability_details(fingerprints)
        raise InvalidMeasurement(
            f"last two measured traces are unstable: {case_id}; "
            f"first_changed_repetition={details['first_changed_repetition']}"
        )


def _assert_exact(results: Iterable[dict[str, Any]], case_id: str) -> None:
    for result in results:
        for measurement in result["measurements"]:
            metric = measurement["metric"]
            tensor = metric.get("tensor", metric)
            if not tensor["exact"] or metric.get("top1_agreement", True) is False:
                raise CampaignError(f"exact gate diverged: {case_id} at {measurement['location']}")


def _flatten_observations(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases:
        result.extend(case.get("repetitions", ()))
    return result


def _flatten_reference_floor(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases:
        for observation in case.get("raw_control_floor", ()):
            if observation.get("pair") != "reference_reference":
                continue
            result.append(
                {
                    "prompt_id": case["prompt_id"],
                    "length_class": case.get("length_class", "legacy"),
                    "exact_prompt_length": case.get("exact_prompt_length"),
                    "mode": ExecutionMode.DECODE_CACHED.value,
                    "point": "logits",
                    "repetition": observation["repetition"],
                    "step": observation["step"],
                    "max_abs_delta": float(
                        observation["metric"]["max_abs_delta"]
                    ),
                    "metric": observation["metric"],
                }
            )
    return result


def _adjudicate_snapshot_candidate(
    snapshot_payload: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    record = next(
        (
            item
            for item in candidate["records"]
            if item["mode"] == ExecutionMode.DECODE_CACHED.value
            and item["point"] == "logits"
            and item["length_class"] == "short"
        ),
        None,
    )
    if record is None:
        raise CampaignError("candidate tolerances omit short cached logits")
    threshold = float(record["max_abs_delta"])
    failures: list[dict[str, Any]] = []
    for observation in snapshot_payload["observations"]:
        for index, comparison in enumerate(observation["comparisons"]):
            if comparison["tensor"]["max_abs_delta"] > threshold:
                failures.append(
                    {
                        "repetition": observation["repetition"],
                        "comparison": index,
                        "metric": comparison,
                    }
                )
            if comparison["top1_agreement"] is False:
                failures.append(
                    {
                        "repetition": observation["repetition"],
                        "comparison": index,
                        "metric": comparison,
                        "reason": "top1_disagreement",
                    }
                )
    return {
        "schema_version": 1,
        "status": "candidate_only",
        "verdict": "CANDIDATE_PASS" if not failures else "FAIL",
        "tolerance_key": ["decode_cached", "logits", "short"],
        "max_abs_delta": threshold,
        "snapshot_stability": snapshot_payload["stability"],
        "first_failure": failures[0] if failures else None,
    }


def _read_case(writer: IncrementalCampaignWriter, case_id: str) -> dict[str, Any]:
    return json.loads((writer.cases_dir / f"{case_id}.json").read_text(encoding="utf-8"))


def _continue_branch(controller: ExecutionStateController, state: Any, tokens: tuple[int, ...], device: Any) -> list[torch.Tensor]:
    controller.activate(state)
    logits: list[torch.Tensor] = []
    for token in tokens:
        inputs = torch.tensor([[token]], dtype=torch.long, device=device)
        outputs = controller.forward(state, input_ids=inputs, use_cache=True)
        logits.append(outputs.logits[0, -1].detach().clone())
    return logits


def _live_cache_storage(cache: Any) -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    for layer in getattr(cache, "layers", ()):
        for attribute in ("conv_states", "recurrent_states", "keys", "values"):
            value = getattr(layer, attribute, None)
            if isinstance(value, torch.Tensor):
                result.add(tensor_storage_identity(value))
    return result


def _branch_storage(state: Any) -> set[tuple[str, int, int]]:
    result = _live_cache_storage(state.cache)
    result.update(
        tensor_storage_identity(value)
        for _, value in iter_snapshot_tensors(state.position)
    )
    result.update(
        tensor_storage_identity(value)
        for _, value in iter_snapshot_tensors(state.model_state)
    )
    return result


def _live_cache_fingerprint(cache: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for index, layer in enumerate(getattr(cache, "layers", ())):
        for attribute in ("conv_states", "recurrent_states", "keys", "values"):
            value = getattr(layer, attribute, None)
            if isinstance(value, torch.Tensor):
                raw = value.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
                digest.update(f"{index}:{attribute}:{tuple(value.shape)}:{value.dtype}".encode())
                digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def _assert_a40_environment() -> None:
    if not torch.cuda.is_available():
        raise CampaignError("SPEC-02 GPU campaign requires CUDA")
    if torch.cuda.device_count() != 1:
        raise CampaignError("SPEC-02 campaign requires exactly one visible A40")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    if name != "NVIDIA A40" or not 40_000_000_000 <= total <= 60_000_000_000:
        raise CampaignError(f"SPEC-02 final gate requires NVIDIA A40 48 GB, got {name!r}/{total}")


def _write_backbone_hash(root: Path, value: BackboneHash) -> None:
    atomic_write_json(root / "preflight" / "backbone_hash.json", value.to_dict())


def _write_failure(root: Path, exc: Exception) -> None:
    try:
        atomic_write_json(
            root / "terminal.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "message": str(exc),
                "exception": type(exc).__name__,
                "finished_at": _now(),
                "pod_action_required": None,
            },
        )
    except Exception:
        pass


def _repo_path(value: str) -> Path:
    return Path(__file__).resolve().parents[3] / value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
