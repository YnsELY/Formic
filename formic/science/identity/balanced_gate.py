"""Balanced cached-decode identity gate for the official campaign.

The stock CUDA path has a measured execution-ordinal effect (ADR-0004).  Raw
left/right traces therefore remain diagnostic, while the blocking wrapper
identity comparison is made between endpoint treatments occupying the same
round-relative ABBA slot.  Every cache is fresh and configured by
``run_schedule_pair`` (A1-A4); no cell or kernel is modified (A11).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import torch

from formic.science.identity.crossover_diagnostic import (
    CPULogitBank,
    PAIR_ENDPOINTS,
    PAIR_NAMES,
)
from formic.science.identity.executor import Endpoint
from formic.science.identity.metrics import compare_logits
from formic.science.identity.schedule_diagnostic import run_schedule_pair


_CONTRASTS = (
    ("left_companion_reference", "reference_reference", "runner_reference", "left"),
    ("left_companion_runner", "reference_runner", "runner_runner", "left"),
    ("right_companion_reference", "reference_reference", "reference_runner", "right"),
    ("right_companion_runner", "runner_reference", "runner_runner", "right"),
)

_NOISE_PAIRS = (
    "reference_reference",
    "runner_runner",
    "reference_runner",
)

# Only the pairs that feed the tolerance floor block the noise phase.  Run
# a40-2026-08-27-r1 measured the mixed reference_runner pair oscillating with
# period 2 under the alternating calendar (repetitions 0 and 2 bit-identical
# at steps 1-5, repetition 1 a different realisation) while RR and NN were
# stable over all three repetitions — a sustained oscillation, not a
# transient, so no burn-in or repetition count can satisfy a raw last-two
# criterion there.  Wrapper identity is decided by the matched ABBA gate; the
# mixed pair remains recorded as a non-blocking diagnostic, the same doctrine
# already applied to raw cross-ordinal fingerprints after crossover r2.
_NOISE_BLOCKING_PAIRS = (
    "reference_reference",
    "runner_runner",
)


def _run_burn_in(
    calendar: str,
    pair_plan: tuple[str, ...],
    endpoints: dict[str, Endpoint],
    *,
    prompt_token_ids: tuple[int, ...],
    forced_token_ids: tuple[int, ...],
    burn_in_pair_traces: int,
    warmup_pair_traces: int,
    process_ordinal_base: int,
) -> tuple[list[dict[str, Any]], int]:
    """Execute measured-path pair traces that are recorded but never admitted.

    Run a40-2026-08-26-r1 measured the documented first-execution realisation
    switch (ADR-0004) extending past a capture-free warmup into the first ~2
    measured pair traces.  The burn-in therefore replays the *exact* measured
    path — ``capture=True`` including the per-step CPU logit copies — so the
    first admitted trace is already in the stationary realisation.  It runs
    only after a non-empty warmup block: without one, the case continues an
    already-hot measured stream and needs no re-entry traces.
    """
    if burn_in_pair_traces < 0:
        raise ValueError("burn-in count must be non-negative")
    executed = burn_in_pair_traces if warmup_pair_traces > 0 else 0
    results: list[dict[str, Any]] = []
    for index in range(executed):
        pair_name = pair_plan[index % len(pair_plan)]
        left_name, right_name = PAIR_ENDPOINTS[pair_name]
        result = run_schedule_pair(
            calendar,
            endpoints[left_name],
            endpoints[right_name],
            prompt_token_ids=prompt_token_ids,
            forced_token_ids=forced_token_ids,
            capture=True,
        )
        if result is None:
            raise RuntimeError("burn-in pair omitted its measured-path result")
        results.append(
            {
                "burn_in_ordinal": index,
                "pair": pair_name,
                "process_lifetime_ordinal_base": process_ordinal_base + index * 16,
                "left_path_fingerprint": result["left_path_fingerprint"],
                "right_path_fingerprint": result["right_path_fingerprint"],
                "cache_independence": result["cache_independence"],
                "autograd_disabled_all_forwards": result[
                    "autograd_disabled_all_forwards"
                ],
                "steps": result["steps"],
            }
        )
    return results, executed


def _burn_in_block(
    requested: int, executed: int, results: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "pair_traces_requested": requested,
        "executed_pair_traces": executed,
        "executed": executed > 0,
        "policy": "measured_discarded_after_warmup",
        "state_capture": True,
        "excluded_from_blocking_criteria": True,
        "pair_results": results,
    }


def run_balanced_logits_gate(
    reference: Endpoint,
    runner: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    forced_token_ids: tuple[int, ...],
    repetitions: int,
    warmup_pair_traces: int,
    burn_in_pair_traces: int,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a four-round Latin ABBA crossover and return matched evidence.

    A fixed quartet would silently label four different configuration ordinals
    as the same slot.  The A40 crossover showed that ordinal position changes
    raw fingerprints, so every pair treatment is rotated through every one of
    the four configuration slots before a contrast is admitted.
    """
    if repetitions < 2:
        raise ValueError("balanced identity gate requires at least two repetitions")
    if warmup_pair_traces < 0:
        raise ValueError("warmup count must be non-negative")
    if len(forced_token_ids) != 8:
        raise ValueError("balanced identity gate is pinned to horizon 8")
    endpoints = {"reference": reference, "runner": runner}
    warmup_plan = tuple(PAIR_NAMES[index % len(PAIR_NAMES)] for index in range(warmup_pair_traces))
    for pair_name in warmup_plan:
        left_name, right_name = PAIR_ENDPOINTS[pair_name]
        result = run_schedule_pair(
            "abba",
            endpoints[left_name],
            endpoints[right_name],
            prompt_token_ids=prompt_token_ids,
            forced_token_ids=forced_token_ids,
            capture=False,
        )
        if result is not None:
            raise RuntimeError("balanced warmup retained a measured result")
    burn_in_results, executed_burn_in = _run_burn_in(
        "abba",
        PAIR_NAMES,
        endpoints,
        prompt_token_ids=prompt_token_ids,
        forced_token_ids=forced_token_ids,
        burn_in_pair_traces=burn_in_pair_traces,
        warmup_pair_traces=warmup_pair_traces,
        process_ordinal_base=warmup_pair_traces * 16,
    )
    burn_in = _burn_in_block(burn_in_pair_traces, executed_burn_in, burn_in_results)

    bank = CPULogitBank()
    pair_results: list[dict[str, Any]] = []
    process_ordinal = (warmup_pair_traces + executed_burn_in) * 16
    try:
        for round_index in range(len(PAIR_NAMES)):
            schedule = PAIR_NAMES[round_index:] + PAIR_NAMES[:round_index]
            for configuration_ordinal, pair_name in enumerate(schedule):
                left_name, right_name = PAIR_ENDPOINTS[pair_name]
                for repetition in range(repetitions):

                    def capture_logits(
                        metadata: dict[str, Any],
                        logits: torch.Tensor,
                        *,
                        current_repetition: int = repetition,
                        current_pair: str = pair_name,
                        current_round: int = round_index,
                        current_configuration: int = configuration_ordinal,
                        process_base: int = process_ordinal,
                    ) -> None:
                        local = int(metadata["pair_local_forward_ordinal"])
                        metadata.update(
                            {
                                "round": current_round,
                                "configuration_ordinal": current_configuration,
                                "pair": current_pair,
                                "repetition": current_repetition,
                                "round_relative_global_forward_ordinal": (
                                    (current_configuration * repetitions)
                                    + current_repetition
                                )
                                * 16
                                + local,
                                "process_lifetime_diagnostic_forward_ordinal": process_base
                                + local,
                            }
                        )
                        bank.add(metadata, logits)

                    result = run_schedule_pair(
                        "abba",
                        endpoints[left_name],
                        endpoints[right_name],
                        prompt_token_ids=prompt_token_ids,
                        forced_token_ids=forced_token_ids,
                        capture=True,
                        cpu_logits_observer=capture_logits,
                    )
                    if result is None:
                        raise RuntimeError("balanced measured pair omitted its result")
                    pair_results.append(
                        {
                            "round": round_index,
                            "configuration_ordinal": configuration_ordinal,
                            "repetition": repetition,
                            "pair": pair_name,
                            "raw_pair_steps": result["steps"],
                            "cache_independence": result["cache_independence"],
                            "autograd_disabled_all_forwards": result[
                                "autograd_disabled_all_forwards"
                            ],
                        }
                    )
                    process_ordinal += 16

            partial = _payload(
                bank,
                pair_results,
                rounds_completed=round_index + 1,
                repetitions_expected=repetitions,
                warmup_pair_traces=warmup_pair_traces,
                burn_in=burn_in,
            )
            if observer is not None:
                observer(partial)

        payload = _payload(
            bank,
            pair_results,
            rounds_completed=len(PAIR_NAMES),
            repetitions_expected=repetitions,
            warmup_pair_traces=warmup_pair_traces,
            burn_in=burn_in,
        )
        if not payload["matched_endpoint_exact"]:
            raise RuntimeError("balanced endpoint identity comparison diverged")
        if not payload["matched_contrast_last_two_exact"]:
            raise RuntimeError("last two balanced contrast traces are unstable")
        if observer is not None:
            observer(payload)
        return payload
    finally:
        bank.clear()


def run_alternating_noise_floor(
    reference: Endpoint,
    runner: Endpoint,
    *,
    prompt_token_ids: tuple[int, ...],
    forced_token_ids: tuple[int, ...],
    repetitions: int,
    warmup_pair_traces: int,
    burn_in_pair_traces: int,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure RR/NN/RN under the stable alternating r6 calendar.

    This is the economical reference-floor control, not the wrapper identity
    verdict. It retains scalar comparisons and path hashes only. The
    mandatory last-two measured-traces assertion blocks on the floor pairs
    (RR and NN); the mixed RN pair is recorded as a non-blocking diagnostic
    (see ``_NOISE_BLOCKING_PAIRS``).
    """
    if repetitions < 2:
        raise ValueError("noise floor requires at least two repetitions")
    if len(forced_token_ids) != 8:
        raise ValueError("noise floor is pinned to horizon 8")
    endpoints = {"reference": reference, "runner": runner}
    warmup_plan = tuple(
        _NOISE_PAIRS[index % len(_NOISE_PAIRS)]
        for index in range(warmup_pair_traces)
    )
    for pair_name in warmup_plan:
        left_name, right_name = PAIR_ENDPOINTS[pair_name]
        if run_schedule_pair(
            "alternating",
            endpoints[left_name],
            endpoints[right_name],
            prompt_token_ids=prompt_token_ids,
            forced_token_ids=forced_token_ids,
            capture=False,
        ) is not None:
            raise RuntimeError("noise-floor warmup retained a measured result")
    burn_in_results, executed_burn_in = _run_burn_in(
        "alternating",
        _NOISE_PAIRS,
        endpoints,
        prompt_token_ids=prompt_token_ids,
        forced_token_ids=forced_token_ids,
        burn_in_pair_traces=burn_in_pair_traces,
        warmup_pair_traces=warmup_pair_traces,
        process_ordinal_base=warmup_pair_traces * 16,
    )
    burn_in = _burn_in_block(burn_in_pair_traces, executed_burn_in, burn_in_results)

    results: list[dict[str, Any]] = []
    for pair_name in _NOISE_PAIRS:
        left_name, right_name = PAIR_ENDPOINTS[pair_name]
        for repetition in range(repetitions):
            result = run_schedule_pair(
                "alternating",
                endpoints[left_name],
                endpoints[right_name],
                prompt_token_ids=prompt_token_ids,
                forced_token_ids=forced_token_ids,
                capture=True,
            )
            if result is None:
                raise RuntimeError("noise-floor measurement omitted its result")
            results.append(
                {
                    "pair": pair_name,
                    "repetition": repetition,
                    "left_path_fingerprint": result["left_path_fingerprint"],
                    "right_path_fingerprint": result["right_path_fingerprint"],
                    "steps": result["steps"],
                    "cache_independence": result["cache_independence"],
                    "autograd_disabled_all_forwards": result[
                        "autograd_disabled_all_forwards"
                    ],
                }
            )
            if observer is not None:
                observer(
                    _noise_payload(
                        results,
                        repetitions=repetitions,
                        warmup_pair_traces=warmup_pair_traces,
                        burn_in=burn_in,
                        complete=False,
                    )
                )
    payload = _noise_payload(
        results,
        repetitions=repetitions,
        warmup_pair_traces=warmup_pair_traces,
        burn_in=burn_in,
        complete=True,
    )
    if not payload["last_two_pair_traces_exact"]:
        raise RuntimeError("noise-floor last two traces are unstable")
    if observer is not None:
        observer(payload)
    return payload


def _noise_payload(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
    warmup_pair_traces: int,
    burn_in: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    stability: dict[str, Any] = {}
    for pair_name in _NOISE_PAIRS:
        pair = [item for item in results if item["pair"] == pair_name]
        stability[pair_name] = (
            len(pair) == repetitions
            and pair[-2]["left_path_fingerprint"]
            == pair[-1]["left_path_fingerprint"]
            and pair[-2]["right_path_fingerprint"]
            == pair[-1]["right_path_fingerprint"]
        )
    floor = [
        {
            "pair": item["pair"],
            "repetition": item["repetition"],
            "step": step["step"],
            "metric": step["comparison"],
        }
        for item in results
        if item["pair"] in _NOISE_BLOCKING_PAIRS
        for step in item["steps"]
    ]
    return {
        "schema_version": 1,
        "protocol": "SPEC-02-alternating-noise-floor-h8-v3",
        "status": "COMPLETE" if complete else "MEASURING",
        "calendar": "alternating",
        "warmup_pair_traces": warmup_pair_traces,
        "warmup_state_capture": False,
        "burn_in": burn_in,
        "repetitions_expected": repetitions,
        "pairs": list(_NOISE_PAIRS),
        "blocking_pairs": list(_NOISE_BLOCKING_PAIRS),
        "pair_results": list(results),
        "pair_stability": stability,
        "last_two_pair_traces_exact": complete
        and all(stability[pair] for pair in _NOISE_BLOCKING_PAIRS),
        "mixed_pair_stability_is_diagnostic_only": True,
        "raw_control_floor": floor,
        "causal_attribution": None,
    }


def _payload(
    bank: CPULogitBank,
    pair_results: list[dict[str, Any]],
    *,
    rounds_completed: int,
    repetitions_expected: int,
    warmup_pair_traces: int,
    burn_in: dict[str, Any],
) -> dict[str, Any]:
    complete = rounds_completed == len(PAIR_NAMES)
    contrasts = _contrasts(bank, repetitions_expected) if complete else []
    signatures = (
        _contrast_signatures(contrasts, repetitions_expected) if complete else []
    )
    last_two_exact = (
        None
        if not complete or repetitions_expected < 2
        else signatures[-2] == signatures[-1]
    )
    controls = _control_floor(pair_results)
    return {
        "schema_version": 1,
        "protocol": "SPEC-02-balanced-abba-latin4-h8-v3",
        "status": "COMPLETE" if complete else "MEASURING",
        "warmup_pair_traces": warmup_pair_traces,
        "warmup_state_capture": False,
        "burn_in": burn_in,
        "rounds_completed": rounds_completed,
        "rounds_expected": len(PAIR_NAMES),
        "repetitions_completed": repetitions_expected if complete else 0,
        "repetitions_expected": repetitions_expected,
        "pair_names": list(PAIR_NAMES),
        "each_pair_covers_configuration_ordinals": list(range(len(PAIR_NAMES))),
        "matched_contrasts": contrasts,
        "matched_endpoint_exact": bool(contrasts)
        and all(item["metric"]["exact"] for item in contrasts),
        "matched_contrast_signatures": signatures,
        "matched_contrast_last_two_exact": last_two_exact,
        "raw_control_floor": controls,
        "raw_pair_results": pair_results,
        "raw_control_stability_is_diagnostic_only": True,
        "causal_attribution": None,
    }


def _contrasts(bank: CPULogitBank, repetitions: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for configuration_ordinal in range(len(PAIR_NAMES)):
        for repetition in range(repetitions):
            for step in range(8):
                for name, reference_pair, candidate_pair, side in _CONTRASTS:
                    reference_meta, reference_logits = bank.get(
                        "abba",
                        configuration_ordinal,
                        reference_pair,
                        repetition,
                        side,
                        step,
                    )
                    candidate_meta, candidate_logits = bank.get(
                        "abba",
                        configuration_ordinal,
                        candidate_pair,
                        repetition,
                        side,
                        step,
                    )
                    evidence = _matched_evidence(
                        reference_meta,
                        reference_logits,
                        candidate_meta,
                        candidate_logits,
                    )
                    results.append(
                        {
                            "contrast": name,
                            "configuration_ordinal": configuration_ordinal,
                            "repetition": repetition,
                            "step": step,
                            "side": side,
                            **evidence,
                        }
                    )
    return results


def _matched_evidence(
    reference_meta: dict[str, Any],
    reference_logits: torch.Tensor,
    candidate_meta: dict[str, Any],
    candidate_logits: torch.Tensor,
) -> dict[str, Any]:
    equal_fields = (
        "calendar",
        "configuration_ordinal",
        "decode_step",
        "side",
        "pair_local_forward_ordinal",
        "round_relative_global_forward_ordinal",
        "repetition",
    )
    mismatches = {
        field: (reference_meta.get(field), candidate_meta.get(field))
        for field in equal_fields
        if reference_meta.get(field) != candidate_meta.get(field)
    }
    if mismatches:
        raise RuntimeError(f"balanced slot metadata mismatch: {mismatches}")
    if reference_meta["process_lifetime_diagnostic_forward_ordinal"] == candidate_meta[
        "process_lifetime_diagnostic_forward_ordinal"
    ]:
        raise RuntimeError("distinct treatment forwards share a process ordinal")
    metric = compare_logits(reference_logits, candidate_logits).to_dict()
    tensor = metric["tensor"]
    return {
        "matching_scope": "case_relative_balanced_crossover",
        "round_relative_calendar_slot_matched": True,
        "round_relative_global_forward_ordinal": reference_meta[
            "round_relative_global_forward_ordinal"
        ],
        "reference_process_lifetime_diagnostic_forward_ordinal": reference_meta[
            "process_lifetime_diagnostic_forward_ordinal"
        ],
        "candidate_process_lifetime_diagnostic_forward_ordinal": candidate_meta[
            "process_lifetime_diagnostic_forward_ordinal"
        ],
        "process_lifetime_diagnostic_forward_ordinals_matched": False,
        "metric": {
            "exact": tensor["exact"],
            "max_abs_delta": tensor["max_abs_delta"],
            "kl_next_token": metric["kl_next_token"],
            "reference_top1": metric["reference_top1"],
            "candidate_top1": metric["candidate_top1"],
            "top1_agreement": metric["top1_agreement"],
            "first_coordinate": tensor["first_coordinate"],
            "reference_value": tensor["reference_value"],
            "candidate_value": tensor["candidate_value"],
        },
    }


def _contrast_signatures(
    contrasts: list[dict[str, Any]], repetitions: int
) -> list[str]:
    signatures: list[str] = []
    for repetition in range(repetitions):
        values = [
            {
                "contrast": item["contrast"],
                "configuration_ordinal": item["configuration_ordinal"],
                "step": item["step"],
                "side": item["side"],
                "metric": _contrast_invariant(item["metric"]),
            }
            for item in contrasts
            if item["repetition"] == repetition
        ]
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        signatures.append(hashlib.sha256(encoded).hexdigest())
    return signatures


def _contrast_invariant(metric: dict[str, Any]) -> dict[str, Any]:
    """Return only endpoint-difference fields, excluding raw token identity."""
    return {
        key: metric.get(key)
        for key in (
            "exact",
            "max_abs_delta",
            "kl_next_token",
            "top1_agreement",
            "first_coordinate",
            "reference_value",
            "candidate_value",
        )
    }


def _control_floor(pair_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for result in pair_results:
        if result["pair"] not in ("reference_reference", "runner_runner"):
            continue
        for step in result["raw_pair_steps"]:
            controls.append(
                {
                    "pair": result["pair"],
                    "repetition": result["repetition"],
                    "step": step["step"],
                    "metric": step["comparison"],
                }
            )
    return controls
