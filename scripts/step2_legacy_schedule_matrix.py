#!/usr/bin/env python3
"""Measure sequential and step-alternated legacy cached-decode schedules.

This is an isolated SPEC-02 diagnostic.  It loads the backbone once, measures
only ``legacy__audit_echo`` at horizon 8, and never changes the campaign gate,
checkpoint, tolerances, or Qwen cells.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MEASURED_REPETITIONS = 3
WARMUP_TRACES = 6
CALENDARS = ("sequential", "alternating")
PAIR_NAMES = ("reference_reference", "runner_runner", "reference_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True, help="new diagnostic directory")
    parser.add_argument(
        "--gpu-max-memory",
        default=None,
        help="optional per-run GPU placement cap, without modifying the config file",
    )
    args = parser.parse_args()

    from dataclasses import replace

    from formic.config.loader import load_config
    from formic.science.determinism import (
        environment_report,
        git_commit,
        prepare_backend_environment,
    )

    config = load_config(args.config)
    if args.gpu_max_memory is not None:
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                max_memory={
                    **config.backbone.max_memory,
                    "0": args.gpu_max_memory,
                },
            ),
        )
    prepare_backend_environment(config.numerics)

    import torch

    from formic.backbone.loader import load_backbone
    from formic.science.identity.artifacts import atomic_write_json
    from formic.science.identity.campaign import _assert_a40_environment
    from formic.science.identity.campaign_plan import timing_continuation
    from formic.science.identity.executor import Endpoint
    from formic.science.identity.memory import IncrementalMemoryWriter
    from formic.science.identity.preflight import release_cuda_working_set
    from formic.science.identity.prompts import load_frozen_corpus
    from formic.science.identity.schedule_diagnostic import run_schedule_pair

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    atomic_write_json(
        output / "run_metadata.json",
        {
            "schema_version": 1,
            "kind": "SPEC-02 legacy schedule matrix diagnostic only",
            "protocol": {
                "prompt_id": "audit_echo",
                "case_id": "legacy__audit_echo",
                "mode": "decode_cached",
                "horizon": 8,
                "warmup_traces_per_shape": WARMUP_TRACES,
                "measured_repetitions_per_configuration": MEASURED_REPETITIONS,
                "capture_profile": "logits_only",
                "continuation": "forced identical timing continuation",
                "one_process_one_load": True,
                "autograd_disabled": True,
                "warmups_capture": False,
                "measured_logits_serialization": "immediate detached CPU hashes and scalars",
                "first_alternating_warmup_memory_instrumentation": True,
                "calendars": list(CALENDARS),
                "pair_configurations": list(PAIR_NAMES),
            },
            "config_sha256": config.config_hash(),
            "placement_override": args.gpu_max_memory,
            "git_commit": git_commit(),
            "environment": environment_report(),
        },
    )
    _assert_a40_environment()

    memory = IncrementalMemoryWriter(output / "memory" / "cuda_memory.json")
    memory.record("before_load")
    handle = None
    matrix: dict[str, Any] = {
        "schema_version": 1,
        "kind": "SPEC-02 legacy schedule matrix diagnostic only",
        "status": "MEASURING",
        "case_id": "legacy__audit_echo",
        "calendars": list(CALENDARS),
        "pair_configurations": list(PAIR_NAMES),
        "configurations": [],
    }
    atomic_write_json(output / "matrix.json", matrix)

    try:
        handle = load_backbone(config)
        memory.record("after_load", handle.model)
        corpus = load_frozen_corpus(REPO_ROOT / config.identity.prompt_set_path)
        corpus.validate_tokenizer(handle.tokenizer)
        prompt = next(
            item
            for item in corpus.prompts
            if item.id == "audit_echo" and item.set_name == "legacy"
        )
        forced = timing_continuation(prompt, config.identity.decode_tokens)
        if len(forced) != 8:
            raise RuntimeError(f"diagnostic requires horizon 8, got {len(forced)}")

        endpoints = {
            "reference": Endpoint("reference", handle.model, handle.view, False),
            "runner": Endpoint("runner", handle.model, handle.view, True),
        }
        pair_endpoints: dict[str, tuple[Endpoint, Endpoint]] = {
            "reference_reference": (endpoints["reference"], endpoints["reference"]),
            "runner_runner": (endpoints["runner"], endpoints["runner"]),
            "reference_runner": (endpoints["reference"], endpoints["runner"]),
        }

        with torch.no_grad():
            for calendar in CALENDARS:
                for pair_name in PAIR_NAMES:
                    left, right = pair_endpoints[pair_name]
                    configuration = {
                        "calendar": calendar,
                        "pair": pair_name,
                        "left_endpoint": left.name,
                        "right_endpoint": right.name,
                        "warmup_paths": 0,
                        "repetitions": [],
                        "stability": {},
                    }
                    matrix["configurations"].append(configuration)
                    _write_configuration(output, matrix, configuration)

                    for warmup in range(WARMUP_TRACES):
                        memory_observer = None
                        if calendar == "alternating" and pair_name == "reference_reference" and warmup == 0:
                            memory_observer = lambda label: memory.record(
                                f"first_alternating_warmup__{label}", handle.model
                            )
                        warmup_result = run_schedule_pair(
                            calendar,
                            left,
                            right,
                            prompt_token_ids=prompt.token_ids,
                            forced_token_ids=forced,
                            capture=False,
                            memory_observer=memory_observer,
                        )
                        if warmup_result is not None:
                            raise RuntimeError("warmup retained a diagnostic result")
                        configuration["warmup_paths"] += 1
                        _write_configuration(output, matrix, configuration)

                    for repetition in range(MEASURED_REPETITIONS):
                        repetition_payload = run_schedule_pair(
                            calendar,
                            left,
                            right,
                            prompt_token_ids=prompt.token_ids,
                            forced_token_ids=forced,
                            capture=True,
                        )
                        if repetition_payload is None:
                            raise RuntimeError("measured repetition omitted its result")
                        repetition_payload["repetition"] = repetition
                        configuration["repetitions"].append(repetition_payload)
                        configuration["stability"] = _stability(configuration["repetitions"])
                        _write_configuration(output, matrix, configuration)
                        memory.record(
                            f"after_{calendar}__{pair_name}__repetition_{repetition}",
                            handle.model,
                        )
                        del repetition_payload
                        gc.collect()

                    # Return inactive transient blocks only after all three
                    # measured repetitions in the configuration are complete.
                    release_cuda_working_set()
                    memory.record(f"after_{calendar}__{pair_name}__cleanup", handle.model)

        matrix["status"] = "COMPLETE"
        matrix["stability"] = _matrix_stability(matrix["configurations"])
        atomic_write_json(output / "matrix.json", matrix)
        memory.record("after_matrix", handle.model)
        memory.write_live_summary(handle.model)
        analysis = _build_analysis(matrix, output)
        atomic_write_json(output / "analysis.json", analysis)
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "message": "LEGACY SCHEDULE MATRIX COMPLETE — NOT AN IDENTITY VERDICT",
                "configurations": len(matrix["configurations"]),
                "stop_pod_before_analysis": True,
            },
        )
        print("LEGACY SCHEDULE MATRIX: PASS")
        print("STOP POD BEFORE ANALYSIS")
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve partial diagnostic evidence
        matrix["status"] = "FAIL"
        matrix["failure"] = {"exception": type(exc).__name__, "message": str(exc)}
        atomic_write_json(output / "matrix.json", matrix)
        if handle is not None:
            try:
                memory.record("on_failure", handle.model)
                memory.write_live_summary(handle.model)
            except Exception:
                pass
        atomic_write_json(
            output / "analysis.json",
            _build_analysis(matrix, output),
        )
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "exception": type(exc).__name__,
                "message": str(exc),
                "stop_pod_before_analysis": True,
            },
        )
        print("LEGACY SCHEDULE MATRIX: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        print("STOP POD BEFORE ANALYSIS")
        return 1
    finally:
        del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _stability(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    def first_changed(field: str) -> int | None:
        return next(
            (
                index
                for index in range(1, len(repetitions))
                if repetitions[index][field] != repetitions[index - 1][field]
            ),
            None,
        )

    def first_changed_step(side: str) -> int | None:
        for index in range(1, len(repetitions)):
            previous = repetitions[index - 1]["steps"]
            current = repetitions[index]["steps"]
            for step in range(min(len(previous), len(current))):
                if previous[step][side]["sha256"] != current[step][side]["sha256"]:
                    return step
        return None

    exact_repetitions = [
        item["repetition"]
        for item in repetitions
        if all(
            step["comparison"]["exact"] and step["comparison"]["top1_agreement"]
            for step in item["steps"]
        )
    ]
    pair_fingerprints = [
        (item["left_path_fingerprint"], item["right_path_fingerprint"])
        for item in repetitions
    ]
    left_changed = first_changed("left_path_fingerprint")
    right_changed = first_changed("right_path_fingerprint")
    changed_repetitions = [value for value in (left_changed, right_changed) if value is not None]
    return {
        "repetitions_measured": len(repetitions),
        "exact_repetitions": exact_repetitions,
        "exact_all_repetitions": len(exact_repetitions) == len(repetitions) == MEASURED_REPETITIONS,
        "last_two_exact": len(pair_fingerprints) >= 2 and pair_fingerprints[-1] == pair_fingerprints[-2],
        "stable_all_repetitions": bool(pair_fingerprints)
        and len(set(pair_fingerprints)) == 1
        and len(pair_fingerprints) == MEASURED_REPETITIONS,
        "first_changed_repetition": min(changed_repetitions)
        if changed_repetitions
        else None,
        "first_changed_repetition_by_side": {
            "left": left_changed,
            "right": right_changed,
        },
        "first_changed_step_by_side": {
            "left": first_changed_step("left"),
            "right": first_changed_step("right"),
        },
        "path_fingerprints_changed": {
            "left": first_changed("left_path_fingerprint") is not None,
            "right": first_changed("right_path_fingerprint") is not None,
        },
        "top1_sequences_changed": {
            "left": _top1_sequences_changed(repetitions, "left"),
            "right": _top1_sequences_changed(repetitions, "right"),
        },
    }


def _top1_sequences_changed(repetitions: list[dict[str, Any]], side: str) -> bool:
    sequences = [tuple(step[side]["top1"] for step in item["steps"]) for item in repetitions]
    return len(set(sequences)) > 1


def _write_configuration(output: Path, matrix: dict[str, Any], configuration: dict[str, Any]) -> None:
    from formic.science.identity.artifacts import atomic_write_json

    filename = f"{configuration['calendar']}__{configuration['pair']}.json"
    atomic_write_json(output / "diagnostics" / filename, configuration)
    atomic_write_json(output / "matrix.json", matrix)


def _matrix_stability(configurations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "configurations_measured": len(configurations),
        "configurations_complete": sum(
            item["stability"].get("repetitions_measured") == MEASURED_REPETITIONS
            for item in configurations
        ),
        "all_configurations_complete": all(
            item["stability"].get("repetitions_measured") == MEASURED_REPETITIONS
            for item in configurations
        ),
    }


def _build_analysis(matrix: dict[str, Any], output: Path) -> dict[str, Any]:
    configurations = matrix.get("configurations", [])
    by_key = {(item["calendar"], item["pair"]): item for item in configurations}
    calendar_effects = []
    for pair in PAIR_NAMES:
        sequential = by_key.get(("sequential", pair))
        alternating = by_key.get(("alternating", pair))
        if sequential is None or alternating is None:
            calendar_effects.append(
                {
                    "pair": pair,
                    "status": "INCOMPLETE",
                    "sequential_present": sequential is not None,
                    "alternating_present": alternating is not None,
                }
            )
            continue
        sequential_complete = (
            sequential["stability"].get("repetitions_measured") == MEASURED_REPETITIONS
        )
        alternating_complete = (
            alternating["stability"].get("repetitions_measured") == MEASURED_REPETITIONS
        )
        if not sequential_complete or not alternating_complete:
            calendar_effects.append(
                {
                    "pair": pair,
                    "status": "INCOMPLETE",
                    "sequential_repetitions_measured": sequential["stability"].get(
                        "repetitions_measured", 0
                    ),
                    "alternating_repetitions_measured": alternating["stability"].get(
                        "repetitions_measured", 0
                    ),
                    "sequential": sequential["stability"],
                    "alternating": alternating["stability"],
                    "interpretation": "No calendar comparison is made until both configurations have all three measured repetitions.",
                }
            )
            continue
        seq_hashes = _side_hashes(sequential)
        alt_hashes = _side_hashes(alternating)
        calendar_effects.append(
            {
                "pair": pair,
                "status": "COMPLETE",
                "direct_observed_hash_change_by_repetition": {
                    "left": [
                        seq_hashes["left"][index] != alt_hashes["left"][index]
                        for index in range(min(len(seq_hashes["left"]), len(alt_hashes["left"])))
                    ],
                    "right": [
                        seq_hashes["right"][index] != alt_hashes["right"][index]
                        for index in range(min(len(seq_hashes["right"]), len(alt_hashes["right"])))
                    ],
                },
                "stability_outcome_changed": sequential["stability"] != alternating["stability"],
                "exact_outcome_changed": sequential["stability"].get("exact_all_repetitions")
                != alternating["stability"].get("exact_all_repetitions"),
                "sequential": sequential["stability"],
                "alternating": alternating["stability"],
                "interpretation": "Observed artifact difference only; this does not identify a cause root.",
            }
        )

    return {
        "schema_version": 1,
        "kind": "SPEC-02 legacy schedule matrix diagnostic analysis",
        "source_directory": str(output),
        "status": matrix.get("status"),
        "case_id": matrix.get("case_id"),
        "protocol": {
            "horizon": 8,
            "warmup_traces_per_shape": WARMUP_TRACES,
            "measured_repetitions_per_configuration": MEASURED_REPETITIONS,
            "calendars": list(CALENDARS),
            "pair_configurations": list(PAIR_NAMES),
            "logits_only": True,
            "no_production_gate_change": True,
        },
        "configurations": [
            {
                "calendar": item["calendar"],
                "pair": item["pair"],
                "warmup_paths": item["warmup_paths"],
                "repetitions_measured": item["stability"].get("repetitions_measured", 0),
                "stable_last_two": item["stability"].get("last_two_exact"),
                "stable_all_repetitions": item["stability"].get("stable_all_repetitions"),
                "exact_all_repetitions": item["stability"].get("exact_all_repetitions"),
                "first_changed_repetition": item["stability"].get("first_changed_repetition"),
                "first_changed_repetition_by_side": item["stability"].get(
                    "first_changed_repetition_by_side", {}
                ),
                "first_changed_step_by_side": item["stability"].get(
                    "first_changed_step_by_side", {}
                ),
                "path_fingerprints_changed": item["stability"].get(
                    "path_fingerprints_changed", {}
                ),
                "top1_sequences_changed": item["stability"].get(
                    "top1_sequences_changed", {}
                ),
                "cache_independence": [
                    repetition["cache_independence"]
                    for repetition in item["repetitions"]
                ],
                "per_repetition_metrics": [
                    {
                        "repetition": repetition["repetition"],
                        "steps": repetition["steps"],
                    }
                    for repetition in item["repetitions"]
                ],
            }
            for item in configurations
        ],
        "calendar_effects": calendar_effects,
        "demonstrated": [
            "The matrix uses one process and one backbone load, with six no-capture warmup traces per configuration and three measured repetitions per configuration.",
            "The sequential calendar records the existing run_aligned_pair order: all left-endpoint steps, then all right-endpoint steps.",
            "The alternating calendar records left endpoint step N followed by right endpoint step N before step N+1.",
            "Each measured repetition records complete logits SHA-256 values, top-1 values, whole-path fingerprints, pair deltas, KL, top-1 agreement, and CUDA cache-independence checks.",
            "A combination is marked exact_all_repetitions only when every recorded logits comparison in all three repetitions is tensor-exact and top-1 agrees.",
            "A combination is marked stable_last_two according to the last-two whole-pair fingerprint rule; stable_all_repetitions requires all three whole-pair fingerprints to match.",
        ],
        "indeterminate": [
            "A difference between calendars is an observed schedule-dependent artifact difference, not a root-cause attribution.",
            "The matrix cannot establish whether any schedule difference is caused by a particular kernel, cache state, model-attached state, allocator state, hardware effect or another factor.",
            "This diagnostic does not justify changing production tolerances, replacing run_aligned_pair, or issuing an identity verdict.",
        ],
    }


def _side_hashes(configuration: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "left": [
            repetition["left_path_fingerprint"]
            for repetition in configuration["repetitions"]
        ],
        "right": [
            repetition["right_path_fingerprint"]
            for repetition in configuration["repetitions"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
