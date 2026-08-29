"""Stock-cell execution paths for the four SPEC-02 identity modes.

This module orchestrates intact Hugging Face CausalLM forwards. It never
reimplements a Qwen cell (A11), never treats ``use_cache=False`` as cache
protection (A1), and every explicit hybrid cache is constructed with the model
config (A2). No rollback or ``crop`` is used (A3/A4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from formic.backbone.groups import HybridGroupView
from formic.backbone.runner import identity_forward
from formic.science.identity.comparison import TraceComparison, compare_forward_traces
from formic.science.identity.protocol import PairTrace
from formic.science.identity.segmentation import segment_slices
from formic.science.identity.trace import ForwardTrace, IdentityTraceCollector
from formic.science.identity.types import CaptureProfile, ExecutionMode, InputShape
from formic.state.snapshot import PositionState


@dataclass(frozen=True)
class Endpoint:
    name: str
    model: Any
    view: HybridGroupView
    through_formic_runner: bool


@dataclass(frozen=True)
class Frame:
    step: int
    shape: InputShape
    trace: ForwardTrace


@dataclass(frozen=True)
class PathTrace:
    endpoint: str
    frames: tuple[Frame, ...]


@dataclass(frozen=True)
class AlignedCasePayload:
    reference: PathTrace
    candidate: PathTrace
    comparisons: tuple[TraceComparison, ...]


def expected_shapes(
    *,
    prompt_length: int,
    mode: ExecutionMode,
    segmentation: str | None,
    decode_steps: int,
) -> tuple[InputShape, ...]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if mode is ExecutionMode.PREFILL_FULL:
        return (InputShape(1, prompt_length, 0),)
    if mode is ExecutionMode.PREFILL_SEGMENTED:
        if segmentation is None:
            raise ValueError("segmented prefill requires a strategy")
        result: list[InputShape] = []
        cached = 0
        for part in segment_slices(prompt_length, segmentation):
            length = part.stop - part.start
            result.append(InputShape(1, length, cached))
            cached += length
        return tuple(result)
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if mode is ExecutionMode.DECODE_CACHED:
        return (InputShape(1, prompt_length, 0),) + tuple(
            InputShape(1, 1, prompt_length + step)
            for step in range(decode_steps - 1)
        )
    if mode is ExecutionMode.DECODE_RECOMPUTE:
        return tuple(
            InputShape(1, prompt_length + step, 0) for step in range(decode_steps)
        )
    raise ValueError(f"unsupported mode {mode}")


def reference_shapes_for_candidate(
    *,
    prompt_length: int,
    candidate_mode: ExecutionMode,
    segmentation: str | None,
    decode_steps: int,
    length_class: str,
) -> tuple[InputShape, ...]:
    """Return the canonical reference shapes for one numerical path comparison.

    Wrapper identity is gated separately by the balanced endpoint crossover.
    Numerical calibration therefore compares segmented prefill with full-prefix
    recomputation and cached decode with full recomputation.  Long cached decode
    retains a cached reference because long recomputation is outside SPEC-02.
    """
    if candidate_mode is ExecutionMode.PREFILL_SEGMENTED:
        if segmentation is None:
            raise ValueError("segmented prefill requires a strategy")
        return tuple(
            InputShape(1, part.stop, 0)
            for part in segment_slices(prompt_length, segmentation)
        )
    if candidate_mode is ExecutionMode.DECODE_CACHED and length_class != "long":
        return expected_shapes(
            prompt_length=prompt_length,
            mode=ExecutionMode.DECODE_RECOMPUTE,
            segmentation=None,
            decode_steps=decode_steps,
        )
    return expected_shapes(
        prompt_length=prompt_length,
        mode=candidate_mode,
        segmentation=segmentation,
        decode_steps=decode_steps,
    )


@torch.no_grad()
def execute_path(
    endpoint: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    mode: ExecutionMode,
    length_class: str,
    segmentation: str | None = None,
    forced_token_ids: tuple[int, ...] = (),
    capture: bool,
    capture_profile: CaptureProfile | None = None,
) -> PathTrace | None:
    if length_class == "long" and mode is ExecutionMode.DECODE_RECOMPUTE:
        raise ValueError("full-recomputation decode is disabled for long prompts")
    device = next(endpoint.model.parameters()).device
    prompt = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    frames: list[Frame] = []

    if mode in (ExecutionMode.PREFILL_FULL, ExecutionMode.PREFILL_SEGMENTED, ExecutionMode.DECODE_CACHED):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=endpoint.model.config)
    else:
        cache = None

    if mode is ExecutionMode.PREFILL_FULL:
        specs = [(0, prompt, cache)]
    elif mode is ExecutionMode.PREFILL_SEGMENTED:
        if segmentation is None:
            raise ValueError("segmented prefill requires a strategy")
        specs = [
            (step, prompt[:, part], cache)
            for step, part in enumerate(segment_slices(len(prompt_token_ids), segmentation))
        ]
    elif mode is ExecutionMode.DECODE_CACHED:
        if not forced_token_ids:
            raise ValueError("cached decode requires a forced continuation")
        specs = [(0, prompt, cache)] + [
            (
                step,
                torch.tensor([[forced_token_ids[step - 1]]], dtype=torch.long, device=device),
                cache,
            )
            for step in range(1, len(forced_token_ids))
        ]
    elif mode is ExecutionMode.DECODE_RECOMPUTE:
        if not forced_token_ids:
            raise ValueError("recompute decode requires a forced continuation")
        specs = [
            (
                step,
                torch.tensor(
                    [prompt_token_ids + forced_token_ids[:step]],
                    dtype=torch.long,
                    device=device,
                ),
                None,
            )
            for step in range(len(forced_token_ids))
        ]
    else:  # pragma: no cover
        raise ValueError(f"unsupported mode {mode}")

    for ordinal, (step, input_ids, active_cache) in enumerate(specs):
        cached_before = int(active_cache.get_seq_length()) if active_cache is not None else 0
        shape = InputShape(1, int(input_ids.shape[-1]), cached_before)
        is_final = ordinal == len(specs) - 1
        # An explicit override (e.g. the logits-only 64-frame probe) applies to
        # every frame; ``None`` keeps the pinned per-class capture rule.
        profile = (
            capture_profile
            if capture_profile is not None
            else _capture_profile(length_class, is_final)
        )
        trace = _forward(
            endpoint,
            input_ids=input_ids,
            cache=active_cache,
            profile=profile,
            capture=capture,
        )
        if capture:
            assert trace is not None
            frames.append(Frame(step, shape, trace))
    if not capture:
        return None
    return PathTrace(endpoint.name, tuple(frames))


@torch.no_grad()
def execute_reference_for_candidate(
    endpoint: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    candidate_mode: ExecutionMode,
    length_class: str,
    segmentation: str | None = None,
    forced_token_ids: tuple[int, ...] = (),
    capture: bool,
    capture_profile: CaptureProfile | None = None,
) -> PathTrace | None:
    """Execute the canonical stock reference for a candidate execution path.

    This is orchestration over intact Hugging Face forwards (A11).  It never
    supplies a cache to a no-cache reference (A1), and all candidate caches are
    still created by :func:`execute_path` with the model config (A2/A3/A4).
    """
    if candidate_mode is ExecutionMode.PREFILL_SEGMENTED:
        if segmentation is None:
            raise ValueError("segmented prefill requires a strategy")
        parts = tuple(segment_slices(len(prompt_token_ids), segmentation))
        frames: list[Frame] = []
        for step, part in enumerate(parts):
            prefix = prompt_token_ids[: part.stop]
            # Each full-prefix reference is a single-frame trace, so left to
            # its own resolution it would always capture as a *final* frame.
            # The comparison pairs it with the candidate's segment frame at
            # the same step, whose profile depends on finality (long class:
            # FINAL_STATE_ONLY only on the last frame, LOGITS_ONLY before).
            # Pin the prefix profile to the matching candidate frame so both
            # sides capture the same registry (run a40-2026-08-28-r2 failed
            # with "model-attached state registry differs" on the first long
            # segmented case without this).
            prefix_profile = (
                capture_profile
                if capture_profile is not None
                else _capture_profile(length_class, step == len(parts) - 1)
            )
            trace = execute_path(
                endpoint,
                prompt_token_ids=prefix,
                mode=ExecutionMode.PREFILL_FULL,
                length_class=length_class,
                capture=capture,
                capture_profile=prefix_profile,
            )
            if capture:
                assert trace is not None and len(trace.frames) == 1
                frame = trace.frames[0]
                frames.append(Frame(step, frame.shape, frame.trace))
        return PathTrace(endpoint.name, tuple(frames)) if capture else None

    reference_mode = (
        ExecutionMode.DECODE_RECOMPUTE
        if candidate_mode is ExecutionMode.DECODE_CACHED and length_class != "long"
        else candidate_mode
    )
    return execute_path(
        endpoint,
        prompt_token_ids=prompt_token_ids,
        mode=reference_mode,
        length_class=length_class,
        segmentation=segmentation,
        forced_token_ids=forced_token_ids,
        capture=capture,
        capture_profile=capture_profile,
    )


def _to_cpu(value: Any) -> Any:
    """Deep-copy a measured trace to CPU so GPU execution storage can die."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if is_dataclass(value):
        return type(value)(
            **{field.name: _to_cpu(getattr(value, field.name)) for field in fields(value)}
        )
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


@torch.no_grad()
def run_cross_path_pair(
    reference: Endpoint,
    candidate: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    candidate_mode: ExecutionMode,
    length_class: str,
    segmentation: str | None = None,
    forced_token_ids: tuple[int, ...] = (),
    capture: bool,
    capture_profile: CaptureProfile | None = None,
) -> PairTrace:
    """Compare a Formic path with its canonical stock numerical reference.

    The endpoint-wrapper identity question is deliberately not answered here;
    it belongs to the balanced crossover gate.  This function measures only
    the cross-path numerical effect required for SPEC-02 tolerances.
    """
    reference_trace = execute_reference_for_candidate(
        reference,
        prompt_token_ids=prompt_token_ids,
        candidate_mode=candidate_mode,
        length_class=length_class,
        segmentation=segmentation,
        forced_token_ids=forced_token_ids,
        capture=capture,
        capture_profile=capture_profile,
    )
    if capture:
        assert reference_trace is not None
        reference_trace = _to_cpu(reference_trace)
    candidate_trace = execute_path(
        candidate,
        prompt_token_ids=prompt_token_ids,
        mode=candidate_mode,
        length_class=length_class,
        segmentation=segmentation,
        forced_token_ids=forced_token_ids,
        capture=capture,
        capture_profile=capture_profile,
    )
    if not capture:
        return PairTrace("", "", None, 0)
    assert reference_trace is not None and candidate_trace is not None
    candidate_trace = _to_cpu(candidate_trace)
    if len(reference_trace.frames) != len(candidate_trace.frames):
        raise RuntimeError("cross-path reference/candidate frame counts differ")
    comparisons = tuple(
        compare_forward_traces(
            ref_frame.trace,
            cand_frame.trace,
            step=cand_frame.step,
            mode=candidate_mode,
            length_class=length_class,
            input_shape=cand_frame.shape,
            allow_cache_applicability_mismatch=True,
            align_reference_hidden_tail=True,
        )
        for ref_frame, cand_frame in zip(reference_trace.frames, candidate_trace.frames)
    )
    payload = AlignedCasePayload(reference_trace, candidate_trace, comparisons)
    ref_fingerprint, ref_count = path_fingerprint(reference_trace)
    cand_fingerprint, cand_count = path_fingerprint(candidate_trace)
    return PairTrace(
        ref_fingerprint,
        cand_fingerprint,
        payload,
        ref_count + cand_count,
    )


def run_aligned_pair(
    reference: Endpoint,
    candidate: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    mode: ExecutionMode,
    length_class: str,
    segmentation: str | None = None,
    forced_token_ids: tuple[int, ...] = (),
    capture: bool,
    capture_profile: CaptureProfile | None = None,
) -> PairTrace:
    ref_trace = execute_path(
        reference,
        prompt_token_ids=prompt_token_ids,
        mode=mode,
        length_class=length_class,
        segmentation=segmentation,
        forced_token_ids=forced_token_ids,
        capture=capture,
        capture_profile=capture_profile,
    )
    cand_trace = execute_path(
        candidate,
        prompt_token_ids=prompt_token_ids,
        mode=mode,
        length_class=length_class,
        segmentation=segmentation,
        forced_token_ids=forced_token_ids,
        capture=capture,
        capture_profile=capture_profile,
    )
    if not capture:
        return PairTrace("", "", None, 0)
    assert ref_trace is not None and cand_trace is not None
    if tuple(frame.shape for frame in ref_trace.frames) != tuple(
        frame.shape for frame in cand_trace.frames
    ):
        raise RuntimeError("aligned endpoints executed different shapes")
    comparisons = tuple(
        compare_forward_traces(
            ref_frame.trace,
            cand_frame.trace,
            step=ref_frame.step,
            mode=mode,
            length_class=length_class,
            input_shape=ref_frame.shape,
        )
        for ref_frame, cand_frame in zip(ref_trace.frames, cand_trace.frames)
    )
    payload = AlignedCasePayload(ref_trace, cand_trace, comparisons)
    ref_fingerprint, ref_count = path_fingerprint(ref_trace)
    cand_fingerprint, cand_count = path_fingerprint(cand_trace)
    return PairTrace(
        ref_fingerprint,
        cand_fingerprint,
        payload,
        ref_count + cand_count,
    )


@torch.no_grad()
def run_greedy_pair(
    reference: Endpoint,
    candidate: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    length_class: str,
    decode_steps: int,
    capture: bool,
) -> PairTrace:
    """Compare paired cached decoding while forcing reference top-1 tokens.

    Unlike sampled decoding, greedy continuation need not be generated in a
    separate pre-pass.  Each reference forward selects the next ID and that
    exact ID is immediately supplied to the candidate at the same decode step.
    This preserves an aligned, interpretable pair without adding forwards to
    the approved campaign budget.
    """
    if decode_steps <= 0:
        raise ValueError("greedy decode requires positive decode_steps")
    from transformers.cache_utils import DynamicCache

    ref_device = next(reference.model.parameters()).device
    cand_device = next(candidate.model.parameters()).device
    reference_cache = DynamicCache(config=reference.model.config)
    candidate_cache = DynamicCache(config=candidate.model.config)
    reference_input = torch.tensor([prompt_token_ids], dtype=torch.long, device=ref_device)
    candidate_input = torch.tensor([prompt_token_ids], dtype=torch.long, device=cand_device)
    ref_frames: list[Frame] = []
    cand_frames: list[Frame] = []
    comparisons: list[TraceComparison] = []
    for step in range(decode_steps):
        ref_before = int(reference_cache.get_seq_length())
        cand_before = int(candidate_cache.get_seq_length())
        if ref_before != cand_before or reference_input.shape != candidate_input.shape:
            raise RuntimeError("greedy endpoints lost aligned cache/input shapes")
        shape = InputShape(1, int(reference_input.shape[-1]), ref_before)
        profile = _capture_profile(length_class, step == decode_steps - 1)
        if capture:
            ref_trace = _forward(
                reference,
                input_ids=reference_input,
                cache=reference_cache,
                profile=profile,
                capture=True,
            )
            assert ref_trace is not None
            next_id = int(torch.argmax(ref_trace.logits).item())
        else:
            outputs = reference.model(
                input_ids=reference_input,
                past_key_values=reference_cache,
                use_cache=True,
            )
            next_id = int(torch.argmax(outputs.logits[0, -1]).item())
        if capture:
            cand_trace = _forward(
                candidate,
                input_ids=candidate_input,
                cache=candidate_cache,
                profile=profile,
                capture=True,
            )
            assert cand_trace is not None
            ref_frames.append(Frame(step, shape, ref_trace))
            cand_frames.append(Frame(step, shape, cand_trace))
            comparisons.append(
                compare_forward_traces(
                    ref_trace,
                    cand_trace,
                    step=step,
                    mode=ExecutionMode.DECODE_CACHED,
                    length_class=length_class,
                    input_shape=shape,
                )
            )
        else:
            if candidate.through_formic_runner:
                identity_forward(
                    SimpleNamespace(model=candidate.model),
                    input_ids=candidate_input,
                    past_key_values=candidate_cache,
                    use_cache=True,
                )
            else:
                candidate.model(
                    input_ids=candidate_input,
                    past_key_values=candidate_cache,
                    use_cache=True,
                )
        reference_input = torch.tensor([[next_id]], dtype=torch.long, device=ref_device)
        candidate_input = torch.tensor([[next_id]], dtype=torch.long, device=cand_device)
    if not capture:
        return PairTrace("", "", None, 0)
    ref_path = PathTrace(reference.name, tuple(ref_frames))
    cand_path = PathTrace(candidate.name, tuple(cand_frames))
    ref_fingerprint, ref_count = path_fingerprint(ref_path)
    cand_fingerprint, cand_count = path_fingerprint(cand_path)
    return PairTrace(
        ref_fingerprint,
        cand_fingerprint,
        AlignedCasePayload(ref_path, cand_path, tuple(comparisons)),
        ref_count + cand_count,
    )


def _capture_profile(length_class: str, is_final: bool) -> CaptureProfile:
    if length_class == "legacy":
        return CaptureProfile.LOGITS_ONLY
    if length_class in ("short", "medium"):
        return CaptureProfile.FULL_BOUNDARIES
    if length_class == "long":
        return CaptureProfile.FINAL_STATE_ONLY if is_final else CaptureProfile.LOGITS_ONLY
    raise ValueError(f"unknown length class {length_class!r}")


def _forward(
    endpoint: Endpoint,
    *,
    input_ids: torch.Tensor,
    cache: Any | None,
    profile: CaptureProfile,
    capture: bool,
) -> ForwardTrace | None:
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": cache is not None}
    if cache is not None:
        kwargs["past_key_values"] = cache
    if not capture:
        if endpoint.through_formic_runner:
            identity_forward(SimpleNamespace(model=endpoint.model), **kwargs)
        else:
            endpoint.model(**kwargs)
        return None

    collector = IdentityTraceCollector(
        model=endpoint.model,
        view=endpoint.view,
        cache=cache,
        capture_profile=profile,
    )
    position = (
        PositionState(sequence_length=int(cache.get_seq_length()) + int(input_ids.shape[-1]))
        if cache is not None
        else None
    )
    if endpoint.through_formic_runner:
        identity_forward(
            SimpleNamespace(model=endpoint.model),
            trace_collector=collector,
            position_state=position,
            **kwargs,
        )
    else:
        with collector:
            outputs = endpoint.model(**kwargs)
        collector.last_trace = collector.finish(outputs, position)
    if collector.last_trace is None:  # pragma: no cover
        raise RuntimeError("trace collector produced no trace")
    return collector.last_trace


def path_fingerprint(value: PathTrace) -> tuple[str, int]:
    """Exact content fingerprint; call only for retained measured traces."""
    count = 0

    def plain(item: Any) -> Any:
        nonlocal count
        if isinstance(item, torch.Tensor):
            count += 1
            raw = item.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
            return {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
                "sha256": hashlib.sha256(memoryview(raw.numpy())).hexdigest(),
            }
        if is_dataclass(item):
            return {field.name: plain(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        if isinstance(item, dict):
            return {str(key): plain(child) for key, child in sorted(item.items())}
        return item

    encoded = json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), count
