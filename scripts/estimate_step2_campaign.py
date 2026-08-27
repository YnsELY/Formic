#!/usr/bin/env python3
"""Print the auditable, weight-free SPEC-02 A40 campaign cost table.

Forward counts and transfer volumes are exact consequences of the proposed
protocol.  Durations are deliberately kept separate: the repository contains
an A40 timing only for a four-token cached trace, not for medium/long shapes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from formic.science.identity.segmentation import segment_slices


WARMUP_TRACES = 6
MEASUREMENT_REPETITIONS = 3
EXACT_GATE_REPETITIONS = 2
ENDPOINTS = 2
DECODE_STEPS = 8
# Protocol v3: measured-then-discarded burn-in after every non-empty warmup
# block (4 pair traces for the paired gates, 1 repetition for cross-path
# cases), and per-endpoint warmups.
BURN_IN_PAIR_TRACES = 4
BURN_IN_REPETITIONS = 1

# BF16 transfer model for the audited 64-layer text backbone.
LOGITS_BYTES = 248_320 * 2
BOUNDARY_HIDDEN_BYTES_PER_TOKEN = 17 * 5_120 * 2
GDN_STATE_BYTES = 79_429_632
KV_BYTES_PER_CACHED_TOKEN = 65_536

# EXP-0008: two explicit HF cached traces, eight forwards each, prompt length 4.
HISTORICAL_SHORT_SECONDS = 49.24790143966675
HISTORICAL_SHORT_FORWARDS = 16
SECONDS_PER_SHORT_FORWARD = HISTORICAL_SHORT_SECONDS / HISTORICAL_SHORT_FORWARDS


@dataclass(frozen=True)
class Prompt:
    id: str
    length_class: str
    tokens: int


@dataclass(frozen=True)
class Cost:
    forwards: int
    input_tokens: int
    transfer_bytes: int
    shapes: str

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.forwards + other.forwards,
            self.input_tokens + other.input_tokens,
            self.transfer_bytes + other.transfer_bytes,
            f"{self.shapes}; {other.shapes}",
        )


PROMPTS = (
    Prompt("short_error_assertion", "short", 26),
    Prompt("short_diff_review", "short", 25),
    Prompt("medium_cache_regression", "medium", 310),
    Prompt("medium_scoped_diff", "medium", 331),
    Prompt("long_resume_incidents", "long", 2_437),
    Prompt("long_monorepo_diff", "long", 2_542),
)


def _full_frame_bytes(input_length: int, cached_after: int) -> int:
    return (
        LOGITS_BYTES
        + BOUNDARY_HIDDEN_BYTES_PER_TOKEN * input_length
        + GDN_STATE_BYTES
        + KV_BYTES_PER_CACHED_TOKEN * cached_after
    )


def _long_final_bytes(cached_after: int) -> int:
    return LOGITS_BYTES + GDN_STATE_BYTES + KV_BYTES_PER_CACHED_TOKEN * cached_after


def _segments(length: int, strategy: str) -> tuple[int, ...]:
    return tuple(part.stop - part.start for part in segment_slices(length, strategy))


def _trace_transfer(
    prompt: Prompt,
    path: str,
    segmentation: str | None,
    decode_steps: int,
) -> int:
    n = prompt.tokens
    is_long = prompt.length_class == "long"
    if path == "prefill_full":
        return _long_final_bytes(n) if is_long else _full_frame_bytes(n, n)
    if path == "prefill_segmented":
        parts = _segments(n, segmentation or "")
        if is_long:
            return len(parts) * LOGITS_BYTES + GDN_STATE_BYTES + KV_BYTES_PER_CACHED_TOKEN * n
        cached = 0
        total = 0
        for part in parts:
            cached += part
            total += _full_frame_bytes(part, cached)
        return total
    if path == "decode_cached":
        if is_long:
            return decode_steps * LOGITS_BYTES + GDN_STATE_BYTES + KV_BYTES_PER_CACHED_TOKEN * (n + decode_steps - 1)
        return _full_frame_bytes(n, n) + sum(
            _full_frame_bytes(1, n + step) for step in range(1, decode_steps)
        )
    if path == "decode_recompute":
        if is_long:
            raise ValueError("long recomputation is outside the campaign")
        return sum(
            LOGITS_BYTES + BOUNDARY_HIDDEN_BYTES_PER_TOKEN * (n + step)
            for step in range(decode_steps)
        )
    raise ValueError(path)


def _frames(
    prompt: Prompt,
    path: str,
    segmentation: str | None,
    decode_steps: int,
) -> tuple[int, ...]:
    n = prompt.tokens
    if path == "prefill_full":
        return (n,)
    if path == "prefill_segmented":
        return _segments(n, segmentation or "")
    if path == "decode_cached":
        return (n,) + (1,) * (decode_steps - 1)
    if path == "decode_recompute":
        return tuple(n + step for step in range(decode_steps))
    raise ValueError(path)


def _reference_frames(
    prompt: Prompt,
    path: str,
    segmentation: str | None,
    decode_steps: int,
) -> tuple[int, ...]:
    if path == "prefill_segmented":
        total = 0
        prefixes = []
        for length in _segments(prompt.tokens, segmentation or ""):
            total += length
            prefixes.append(total)
        return tuple(prefixes)
    if path == "decode_cached" and prompt.length_class != "long":
        return tuple(prompt.tokens + step for step in range(decode_steps))
    return _frames(prompt, path, segmentation, decode_steps)


def _reference_transfer(
    prompt: Prompt,
    path: str,
    segmentation: str | None,
    decode_steps: int,
) -> int:
    if path == "prefill_segmented":
        prefixes = _reference_frames(prompt, path, segmentation, decode_steps)
        if prompt.length_class == "long":
            return sum(_long_final_bytes(prefix) for prefix in prefixes)
        return sum(_full_frame_bytes(prefix, prefix) for prefix in prefixes)
    if path == "decode_cached" and prompt.length_class != "long":
        return _trace_transfer(prompt, "decode_recompute", None, decode_steps)
    return _trace_transfer(prompt, path, segmentation, decode_steps)


def main_case(
    prompt: Prompt,
    path: str,
    segmentation: str | None = None,
    *,
    decode_steps: int = DECODE_STEPS,
) -> Cost:
    frames = _frames(prompt, path, segmentation, decode_steps)
    reference_frames = _reference_frames(prompt, path, segmentation, decode_steps)
    # Greedy and sampled decoding share the same warm traces and burn-in.
    # Prefill has one variant. Every measured comparison has two aligned
    # endpoints.
    variants = 2 if path.startswith("decode_") else 1
    measured_pairs = MEASUREMENT_REPETITIONS * variants
    # Per-endpoint warmup ledgers: both endpoints warm their own path, except
    # the short/medium cached candidate whose recompute reference shapes were
    # already warmed by the preceding decode_recompute case.
    if path == "decode_cached" and prompt.length_class != "long":
        reference_warmups, candidate_warmups = 0, WARMUP_TRACES
    else:
        reference_warmups, candidate_warmups = WARMUP_TRACES, WARMUP_TRACES
    # One measured-then-discarded pair runs after the (always non-empty)
    # warmup block of each case, shared by the decode variants.
    burn_in_pairs = BURN_IN_REPETITIONS
    return Cost(
        forwards=(
            reference_warmups * len(reference_frames)
            + candidate_warmups * len(frames)
            + (burn_in_pairs + measured_pairs)
            * (len(reference_frames) + len(frames))
        ),
        input_tokens=(
            reference_warmups * sum(reference_frames)
            + candidate_warmups * sum(frames)
            + (burn_in_pairs + measured_pairs)
            * (sum(reference_frames) + sum(frames))
        ),
        transfer_bytes=(burn_in_pairs + measured_pairs)
        * (
            _reference_transfer(prompt, path, segmentation, decode_steps)
            + _trace_transfer(prompt, path, segmentation, decode_steps)
        ),
        shapes="/".join(str(value) for value in frames),
    )


def _aggregate(prompts: tuple[Prompt, ...], path: str, segmentation: str | None = None) -> Cost:
    costs = [main_case(prompt, path, segmentation) for prompt in prompts]
    result = costs[0]
    for cost in costs[1:]:
        result += cost
    return result


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f}"


def _short_projection_hours(forwards: int) -> str:
    return f"{forwards * SECONDS_PER_SHORT_FORWARD / 3600:.2f}"


def _print_main_table() -> tuple[int, int, int]:
    print("| Classe | Chemin | Segmentation | Formes (tokens d'entrée/forward) | Forwards | Transfert GiB | Équiv. EXP-0008 (h, non calibré) | Durée par classe |")
    print("|---|---|---|---|---:|---:|---:|---|")
    totals = Cost(0, 0, 0, "")
    for length_class in ("short", "medium", "long"):
        prompts = tuple(prompt for prompt in PROMPTS if prompt.length_class == length_class)
        cases: list[tuple[str, str | None]] = [("prefill_full", None)]
        segmentations = ("median", "quarters") if length_class == "long" else (
            "early", "median", "late", "quarters"
        )
        cases.extend(("prefill_segmented", item) for item in segmentations)
        cases.append(("decode_cached", None))
        if length_class != "long":
            cases.append(("decode_recompute", None))
        for path, segmentation in cases:
            selected = prompts
            if path.startswith("decode_"):
                selected = tuple(
                    prompt
                    for prompt in prompts
                    if prompt.id
                    in {
                        "short_error_assertion",
                        "medium_cache_regression",
                        "long_resume_incidents",
                    }
                )
            cost = _aggregate(selected, path, segmentation)
            totals += cost
            shape = cost.shapes.replace("; ", "<br>")
            print(
                f"| {length_class} | `{path}` | {segmentation or '—'} | {shape} | "
                f"{cost.forwards:,} | {_gib(cost.transfer_bytes)} | "
                f"{_short_projection_hours(cost.forwards)} | N/C |"
            )
    print(
        f"| **TOTAL principal** | | | | **{totals.forwards:,}** | "
        f"**{_gib(totals.transfer_bytes)}** | **{_short_projection_hours(totals.forwards)}** | **N/C** |"
    )
    return totals.forwards, totals.input_tokens, totals.transfer_bytes


def _phase_costs(main_totals: tuple[int, int, int]) -> list[tuple[str, Cost, str]]:
    main = Cost(*main_totals, "voir tableau principal")
    legacy = tuple(Prompt(f"legacy_{i + 1}", "legacy", n) for i, n in enumerate((4, 5, 20, 24, 86, 86)))
    all_prompts = legacy + PROMPTS

    # Inertness: six warm traces, one measured-then-discarded OFF/ON burn-in
    # pair, then two OFF and two ON exact traces.
    inert_forwards = sum(12 for _ in all_prompts)
    inert_tokens = sum(12 * prompt.tokens for prompt in all_prompts)
    inert_transfer = sum(
        3 * (_long_final_bytes(prompt.tokens) if prompt.length_class == "long" else _full_frame_bytes(prompt.tokens, prompt.tokens))
        + 3 * LOGITS_BYTES
        for prompt in all_prompts
    )
    inert = Cost(inert_forwards, inert_tokens, inert_transfer, "prefill corpus complet")

    # Legacy gate: every treatment rotates through all four configuration
    # ordinals, with two measured repetitions per slot. Six pair warmups are
    # shared by exact path shape (five distinct legacy lengths) and each
    # non-empty warmup block is followed by four measured-then-discarded
    # burn-in pair traces.
    unique_legacy = tuple({prompt.tokens: prompt for prompt in legacy}.values())
    legacy_measured_pair_traces = len(legacy) * 4 * 4 * EXACT_GATE_REPETITIONS
    legacy_warm_pair_traces = len(unique_legacy) * WARMUP_TRACES
    legacy_burn_in_pair_traces = len(unique_legacy) * BURN_IN_PAIR_TRACES
    legacy_cost = Cost(
        forwards=(
            legacy_measured_pair_traces
            + legacy_warm_pair_traces
            + legacy_burn_in_pair_traces
        )
        * 16,
        input_tokens=(
            sum(
                4 * 4 * EXACT_GATE_REPETITIONS * 2 * (prompt.tokens + DECODE_STEPS - 1)
                for prompt in legacy
            )
            + sum(
                (WARMUP_TRACES + BURN_IN_PAIR_TRACES)
                * 2
                * (prompt.tokens + DECODE_STEPS - 1)
                for prompt in unique_legacy
            )
        ),
        transfer_bytes=(
            legacy_measured_pair_traces + legacy_burn_in_pair_traces
        )
        * 16
        * LOGITS_BYTES,
        shapes="ABBA Latin-4, cached 8 étapes",
    )

    # Alternating r6 noise floor: RR/NN/RN × 3 repetitions. Audit_echo reuses
    # the legacy warmup (no warmup, hence no burn-in); short and medium each
    # warm six pair traces then burn in four measured-then-discarded pairs.
    noise_prompts = (legacy[0], PROMPTS[0], PROMPTS[2])
    noise_pair_traces = (
        9,
        WARMUP_TRACES + BURN_IN_PAIR_TRACES + 9,
        WARMUP_TRACES + BURN_IN_PAIR_TRACES + 9,
    )
    noise_captured_pair_traces = (9, BURN_IN_PAIR_TRACES + 9, BURN_IN_PAIR_TRACES + 9)
    noise_cost = Cost(
        forwards=sum(noise_pair_traces) * 16,
        input_tokens=sum(
            traces * 2 * (prompt.tokens + DECODE_STEPS - 1)
            for prompt, traces in zip(noise_prompts, noise_pair_traces)
        ),
        transfer_bytes=sum(noise_captured_pair_traces) * 16 * LOGITS_BYTES,
        shapes="alternating, cached 8 étapes, logits seulement",
    )

    # One reference sampled continuation per seed, then token IDs are frozen.
    continuation_prompts = (PROMPTS[0], PROMPTS[2], PROMPTS[4])
    continuation = Cost(
        forwards=len(continuation_prompts) * 4 * DECODE_STEPS,
        input_tokens=sum(4 * (prompt.tokens + DECODE_STEPS - 1) for prompt in continuation_prompts),
        transfer_bytes=len(continuation_prompts) * 4 * DECODE_STEPS * 8,
        shapes="greedy + 3 seeds, génération de référence uniquement",
    )

    # Snapshot/restore: one measured-then-discarded burn-in repetition plus
    # three measured repetitions (16 forwards each).
    snapshot_prompt = Prompt("audit_echo", "short", 4)
    snapshot_transfer = 8 * _trace_transfer(
        snapshot_prompt, "decode_cached", None, DECODE_STEPS
    )
    snapshot_validation = Cost(
        forwards=8 * DECODE_STEPS,
        input_tokens=8 * (snapshot_prompt.tokens + DECODE_STEPS - 1),
        transfer_bytes=snapshot_transfer,
        shapes="continuité cache 8 étapes, 3 comparaisons alignées + burn-in",
    )

    # The already proposed 64-step accumulation probe (short + medium):
    # 6+6 warm traces, one burn-in pair and three measured pairs per prompt.
    accumulation_prompts = (PROMPTS[0], PROMPTS[2])
    accumulation = Cost(
        forwards=len(accumulation_prompts) * 20 * 64,
        input_tokens=sum(20 * (prompt.tokens + 63) for prompt in accumulation_prompts),
        transfer_bytes=len(accumulation_prompts) * 8 * 64 * LOGITS_BYTES,
        shapes="cached vs recompute, 64 étapes, logits seulement",
    )
    preflight = Cost(
        forwards=333,
        input_tokens=0,
        transfer_bytes=0,
        shapes="18 chemins candidats + 12 compagnons canoniques distincts, 1 dry + 2 chronométrés",
    )
    return [
        ("Préflight chronométré", preflight, "budget automatique après écriture"),
        ("Gate inertie trace", inert, "bloquante, corpus complet"),
        ("Continuité legacy", legacy_cost, "exacte, hors calibration"),
        ("Plancher de bruit", noise_cost, "3 prompts, calendrier partagé"),
        ("Snapshot/restore réel", snapshot_validation, "continuité, adjudication après tolérances"),
        ("Continuations échantillonnées", continuation, "seul usage des 3 seeds"),
        ("Calibration principale", main, "tableau ci-dessus"),
        ("Sonde accumulation 64", accumulation, "diagnostic obligatoire, logits seulement"),
    ]


def main() -> None:
    print("## Calibration principale — option B, corpus gelé\n")
    main_totals = _print_main_table()
    print("\n## Session complète\n")
    print("| Phase | Forwards | Tokens d'entrée cumulés | Transfert GiB | Équiv. EXP-0008 (h, non calibré) | Note |")
    print("|---|---:|---:|---:|---:|---|")
    phases = _phase_costs(main_totals)
    total = Cost(0, 0, 0, "")
    for name, cost, note in phases:
        total += cost
        print(
            f"| {name} | {cost.forwards:,} | {cost.input_tokens:,} | {_gib(cost.transfer_bytes)} | "
            f"{_short_projection_hours(cost.forwards)} | {note} |"
        )
    print(
        f"| **TOTAL planifié** | **{total.forwards:,}** | **{total.input_tokens:,}** | "
        f"**{_gib(total.transfer_bytes)}** | **{_short_projection_hours(total.forwards)}** | |"
    )
    print(
        f"\nAncre EXP-0008: {SECONDS_PER_SHORT_FORWARD:.6f} s/forward caché à préfill 4 tokens. "
        "Cette colonne de sensibilité n'est une calibration d'aucune des trois bandes SPEC-02."
    )


if __name__ == "__main__":
    main()
