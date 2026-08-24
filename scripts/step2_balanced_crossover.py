#!/usr/bin/env python3
"""Run the isolated balanced ABBA/BAAB SPEC-02 crossover diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WARMUP_PLAN = (
    ("abba", "reference_reference"),
    ("baab", "runner_runner"),
    ("abba", "reference_runner"),
    ("baab", "runner_reference"),
    ("abba", "runner_runner"),
    ("baab", "reference_reference"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-max-memory", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import (
        environment_report,
        git_commit,
        git_dirty,
        prepare_backend_environment,
    )

    output = Path(args.output)
    if args.resume:
        if not output.is_dir() or not (output / "manifest.json").is_file():
            raise SystemExit("--resume requires an existing crossover manifest")
        _prevalidate_manifest(output / "manifest.json")
    elif output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    if git_dirty(REPO_ROOT) is not False:
        raise SystemExit("refusing measurement unless Git is available and clean")
    commit = git_commit(REPO_ROOT)
    if not commit:
        raise SystemExit("unable to resolve Git commit")

    config = load_config(args.config)
    if args.gpu_max_memory is not None:
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                max_memory={**config.backbone.max_memory, "0": args.gpu_max_memory},
            ),
        )
    if not config.identity_mode():
        raise SystemExit("crossover diagnostic requires all Formic behavior disabled")
    prepare_backend_environment(config.numerics)

    import torch

    from formic.backbone.inventory import CheckpointInventory
    from formic.backbone.loader import load_backbone
    from formic.science.backbone_hash import canonical_backbone_hash
    from formic.science.identity.artifacts import ArtifactError, atomic_write_json
    from formic.science.identity.campaign import _assert_a40_environment
    from formic.science.identity.campaign_plan import timing_continuation
    from formic.science.identity.crossover_diagnostic import (
        CrossoverBlocked,
        CrossoverIdentity,
        CrossoverWriter,
        CPULogitBank,
        AttemptMemoryWriter,
        MEASURED_REPETITIONS,
        PAIR_ENDPOINTS,
        assert_resumable_terminal,
        balanced_design,
        build_analysis,
        build_inversion_checks,
        build_ordinal_position_observations,
        build_same_slot_contrasts,
        prepare_attempt_metadata,
        process_lifetime_diagnostic_forward_ordinal,
        round_configurations,
        round_relative_global_forward_ordinal,
        stability_against_prior,
        validate_balanced_design,
    )
    from formic.science.identity.executor import Endpoint
    from formic.science.identity.preflight import release_cuda_working_set
    from formic.science.identity.prompts import load_frozen_corpus
    from formic.science.identity.schedule_diagnostic import run_schedule_pair

    if args.resume:
        assert_resumable_terminal(output / "terminal.json")
    _assert_a40_environment()
    if not args.resume:
        output.mkdir(parents=True)
    corpus = load_frozen_corpus(REPO_ROOT / config.identity.prompt_set_path)
    inventory = CheckpointInventory.from_checkpoint(config.backbone.checkpoint_path)
    inventory.validate_against_audit()
    backbone = canonical_backbone_hash(inventory)
    identity = CrossoverIdentity(
        protocol="SPEC-02-balanced-crossover-h8-v3",
        config_sha256=config.config_hash(),
        corpus_sha256=corpus.corpus_sha256,
        corpus_source_sha256=corpus.source_sha256,
        git_commit=commit,
        backbone_sha256=backbone.sha256,
    )
    writer = CrossoverWriter(output, identity)
    writer.validate()
    design = balanced_design()
    design_validation = validate_balanced_design(design)
    atomic_write_json(
        output / "design.json",
        {
            "schema_version": 1,
            "ordinal_name": "round_relative_global_forward_ordinal",
            "ordinal_formula": (
                "((configuration_ordinal * measured_repetitions) + repetition) "
                "* forwards_per_pair + pair_local_forward_ordinal"
            ),
            "ordinal_range": [0, 383],
            "process_lifetime_diagnostic_forward_ordinal_formula": (
                "96 + round * 384 + round_relative_global_forward_ordinal"
            ),
            "process_lifetime_measured_range_per_attempt": [96, 3167],
            "process_lifetime_scope": (
                "diagnostic model forwards within one attempt; excludes load internals"
            ),
            "validation": design_validation,
            "configurations": [item.to_dict() for item in design],
        },
    )
    metadata_base: dict[str, Any] = {
        "schema_version": 1,
        "kind": "balanced SPEC-02 crossover diagnostic only",
        "config_sha256": config.config_hash(),
        "corpus_sha256": corpus.corpus_sha256,
        "corpus_source_sha256": corpus.source_sha256,
        "git_commit": commit,
        "canonical_backbone": backbone.to_dict(),
        "protocol": {
            "prompt_id": "audit_echo",
            "mode": "decode_cached",
            "horizon": 8,
            "calendars": ["abba", "baab"],
            "pairs": list(PAIR_ENDPOINTS),
            "measured_rounds": 8,
            "scheduled_occurrences": 64,
            "measured_repetitions_per_scheduled_occurrence": MEASURED_REPETITIONS,
            "measured_pair_traces": 64 * MEASURED_REPETITIONS,
            "shared_no_capture_warmup_pair_traces": len(WARMUP_PLAN),
            "ordinal_name": "round_relative_global_forward_ordinal",
            "ordinal_formula": (
                "((configuration_ordinal * 3) + repetition) * 16 "
                "+ pair_local_forward_ordinal"
            ),
            "ordinal_range_per_round": [0, 383],
            "process_lifetime_diagnostic_forward_ordinal_formula": (
                "96 + round * 384 + round_relative_global_forward_ordinal"
            ),
            "process_lifetime_measured_range_per_attempt": [96, 3167],
            "process_lifetime_counts_diagnostic_model_forwards_not_load_internals": True,
            "one_process_one_load": True,
            "not_an_identity_verdict": True,
        },
    }
    metadata, attempt_id = prepare_attempt_metadata(
        output / "run_metadata.json",
        metadata_base,
        {
            "resume": args.resume,
            "placement_override": args.gpu_max_memory,
            "environment": environment_report(),
        },
    )
    memory = AttemptMemoryWriter(output / "cuda_memory.json", attempt_id)
    live_summaries = _existing_live_summaries(output / "live_tensors.json")
    bank = CPULogitBank()
    handle = None
    configurations: list[dict[str, Any]] = []
    matrix: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "ordinal_name": "round_relative_global_forward_ordinal",
        "configurations": [],
    }
    atomic_write_json(output / "matrix.json", matrix)
    try:
        memory.record("before_load")
        # Exactly one production backbone load is allowed in this process.
        handle = load_backbone(config)
        memory.record("after_load", handle.model)
        corpus.validate_tokenizer(handle.tokenizer)

        prompt = next(
            item
            for item in corpus.prompts
            if item.id == "audit_echo" and item.set_name == "legacy"
        )
        forced = timing_continuation(prompt, 8)
        if len(forced) != 8:
            raise CrossoverBlocked(f"crossover requires horizon 8, got {len(forced)}")
        endpoints = {
            "reference": Endpoint("reference", handle.model, handle.view, False),
            "runner": Endpoint("runner", handle.model, handle.view, True),
        }
        pairs = {
            pair: (endpoints[left], endpoints[right])
            for pair, (left, right) in PAIR_ENDPOINTS.items()
        }

        memory.record("before_warmups", handle.model)
        with torch.no_grad():
            for calendar, pair in WARMUP_PLAN:
                left, right = pairs[pair]
                result = run_schedule_pair(
                    calendar,
                    left,
                    right,
                    prompt_token_ids=prompt.token_ids,
                    forced_token_ids=forced,
                    capture=False,
                )
                if result is not None:
                    raise CrossoverBlocked("a no-capture warmup retained a result")
        memory.record("after_warmups", handle.model)
        release_cuda_working_set()
        memory.record("after_warmups_cleanup", handle.model)
        _check_live_execution_storage(
            handle.model, output, live_summaries, "after_warmups"
        )

        for round_index in range(8):
            memory.record(f"before_round_{round_index}", handle.model)
            round_payloads: list[dict[str, Any]] = []
            for configuration in round_configurations(round_index):
                left, right = pairs[configuration.pair]
                repetitions: list[dict[str, Any]] = []
                all_stability_exact = True
                for repetition in range(MEASURED_REPETITIONS):
                    def observe_cpu_logits(
                        record: dict[str, Any],
                        logits: torch.Tensor,
                        *,
                        current_repetition: int = repetition,
                    ) -> None:
                        record.update(
                            {
                                "round": round_index,
                                "configuration_ordinal": configuration.configuration_ordinal,
                                "pair": configuration.pair,
                                "repetition": current_repetition,
                                "round_relative_global_forward_ordinal": (
                                    round_relative_global_forward_ordinal(
                                        configuration.configuration_ordinal,
                                        current_repetition,
                                        record["pair_local_forward_ordinal"],
                                    )
                                ),
                                "process_lifetime_diagnostic_forward_ordinal": (
                                    process_lifetime_diagnostic_forward_ordinal(
                                        round_index,
                                        configuration.configuration_ordinal,
                                        current_repetition,
                                        record["pair_local_forward_ordinal"],
                                    )
                                ),
                            }
                        )
                        bank.add(record, logits)

                    result = run_schedule_pair(
                        configuration.calendar,
                        left,
                        right,
                        prompt_token_ids=prompt.token_ids,
                        forced_token_ids=forced,
                        capture=True,
                        cpu_logits_observer=observe_cpu_logits,
                    )
                    if result is None:
                        raise CrossoverBlocked("measured crossover repetition omitted its result")
                    _enrich_result(result, configuration, repetition)
                    slots = []
                    for step in result["steps"]:
                        for side in ("left", "right"):
                            record = step[side]
                            current_meta, current_logits = bank.get(
                                configuration.calendar,
                                configuration.configuration_ordinal,
                                configuration.pair,
                                repetition,
                                side,
                                step["step"],
                            )
                            if repetition:
                                prior_meta, prior_logits = bank.get(
                                    configuration.calendar,
                                    configuration.configuration_ordinal,
                                    configuration.pair,
                                    repetition - 1,
                                    side,
                                    step["step"],
                                )
                            else:
                                prior_meta = prior_logits = None
                            stability = stability_against_prior(
                                current_meta,
                                current_logits,
                                prior_meta,
                                prior_logits,
                            )
                            if stability is not None:
                                all_stability_exact = all_stability_exact and stability["exact"]
                            slots.append(
                                {
                                    **record,
                                    "pair_comparison": step["comparison"],
                                    "stability_versus_prior_repetition": stability,
                                }
                            )
                    repetitions.append(
                        {
                            "repetition": repetition,
                            "forward_order": result["forward_order"],
                            "cache_independence": result["cache_independence"],
                            "autograd_disabled_all_forwards": result[
                                "autograd_disabled_all_forwards"
                            ],
                            "slots": slots,
                        }
                    )
                    partial = _configuration_payload(
                        configuration,
                        repetitions,
                        all_stability_exact=all_stability_exact,
                        final_last_two=[],
                        status="BLOCKED",
                    )
                    writer.write_diagnostic(configuration.checkpoint_id, partial)
                    matrix["active_configuration"] = configuration.to_dict()
                    matrix["active_repetitions"] = len(repetitions)
                    atomic_write_json(output / "matrix.json", matrix)
                    del result

                final_last_two = _last_two_checks(bank, configuration)
                payload = _configuration_payload(
                    configuration,
                    repetitions,
                    all_stability_exact=all_stability_exact,
                    final_last_two=final_last_two,
                    status="COMPLETE",
                )
                writer.write_diagnostic(configuration.checkpoint_id, payload)
                writer.write_configuration(configuration.checkpoint_id, payload)
                configurations.append(payload)
                round_payloads.append(payload)
                matrix["configurations"] = [
                    _configuration_summary(item) for item in configurations
                ]
                matrix.pop("active_configuration", None)
                matrix.pop("active_repetitions", None)
                atomic_write_json(output / "matrix.json", matrix)

            writer.write_round(
                round_index,
                {
                    "schema_version": 1,
                    "round": round_index,
                    "ordinal_name": "round_relative_global_forward_ordinal",
                    "configurations": round_payloads,
                },
            )
            memory.record(f"after_round_{round_index}", handle.model)
            release_cuda_working_set()
            memory.record(f"after_round_{round_index}_cleanup", handle.model)
            _check_live_execution_storage(
                handle.model, output, live_summaries, f"after_round_{round_index}"
            )

        writer.validate()
        memory.record("before_analysis", handle.model)
        same_slot = build_same_slot_contrasts(bank)
        inversions = build_inversion_checks(bank)
        ordinal_positions = build_ordinal_position_observations(configurations)
        atomic_write_json(
            output / "same_slot_contrasts.json",
            {"schema_version": 1, "contrasts": same_slot},
        )
        atomic_write_json(
            output / "inversion_checks.json",
            {"schema_version": 1, "checks": inversions},
        )
        atomic_write_json(
            output / "ordinal_position_observations.json",
            ordinal_positions,
        )
        analysis = build_analysis(
            configurations=configurations,
            same_slot_contrasts=same_slot,
            inversion_checks=inversions,
            design_validation=design_validation,
        )
        atomic_write_json(output / "analysis.json", analysis)
        memory.record("after_analysis", handle.model)
        _check_live_execution_storage(
            handle.model, output, live_summaries, "after_analysis"
        )
        memory.record("before_cleanup", handle.model)
        bank.clear()
        gc.collect()
        memory.record("after_cleanup", handle.model)
        _check_live_execution_storage(
            handle.model, output, live_summaries, "after_cleanup"
        )
        matrix["status"] = analysis["status"]
        matrix["analysis_checks"] = analysis["checks"]
        matrix["campaign_readiness"] = analysis["readiness"]["status"]
        atomic_write_json(output / "matrix.json", matrix)
        terminal_status = analysis["status"]
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": terminal_status,
                "message": (
                    "BALANCED CROSSOVER COMPLETE; THIS IS NOT A SPEC-02 IDENTITY VERDICT"
                    if terminal_status == "COMPLETE"
                    else "BALANCED CROSSOVER BLOCKED; THIS IS NOT A SPEC-02 IDENTITY VERDICT"
                ),
                "campaign_readiness": analysis["readiness"]["status"],
                "ready_for_full_spec_02_campaign": analysis["readiness"][
                    "ready_for_full_spec_02_campaign"
                ],
            },
        )
        print(f"BALANCED CROSSOVER: {terminal_status}")
        print("THIS IS NOT A SPEC-02 IDENTITY VERDICT")
        return 0 if terminal_status == "COMPLETE" else 1
    except Exception as exc:  # noqa: BLE001 - partial diagnostic evidence is mandatory
        status = "BLOCKED" if isinstance(exc, (CrossoverBlocked, ArtifactError)) else "FAIL"
        matrix["status"] = status
        matrix["failure"] = {"exception": type(exc).__name__, "message": str(exc)}
        atomic_write_json(output / "matrix.json", matrix)
        fallback = build_analysis(
            configurations=configurations,
            same_slot_contrasts=[],
            inversion_checks=[],
            design_validation=design_validation,
        )
        fallback["status"] = status
        fallback["failure"] = matrix["failure"]
        atomic_write_json(output / "analysis.json", fallback)
        if handle is not None:
            try:
                memory.record("on_failure", handle.model)
                _write_live_summary(handle.model, output, live_summaries, "on_failure")
            except Exception:
                pass
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": status,
                "exception": type(exc).__name__,
                "message": f"{exc}; THIS IS NOT A SPEC-02 IDENTITY VERDICT",
            },
        )
        print(f"BALANCED CROSSOVER: {status}")
        print(f"  {type(exc).__name__}: {exc}")
        print("THIS IS NOT A SPEC-02 IDENTITY VERDICT")
        return 1
    finally:
        bank.clear()
        configurations.clear()
        gc.collect()
        # No allocator reset occurs here: failures preserve their allocation
        # state, and successful releases happen only at whole-phase boundaries.
        del handle


def _enrich_result(result: dict[str, Any], configuration: Any, repetition: int) -> None:
    from formic.science.identity.crossover_diagnostic import (
        process_lifetime_diagnostic_forward_ordinal,
        round_relative_global_forward_ordinal,
    )

    shared = {
        "round": configuration.round,
        "configuration_ordinal": configuration.configuration_ordinal,
        "pair": configuration.pair,
        "repetition": repetition,
    }
    for record in result["forward_order"]:
        record.update(shared)
        record["round_relative_global_forward_ordinal"] = round_relative_global_forward_ordinal(
            configuration.configuration_ordinal,
            repetition,
            record["pair_local_forward_ordinal"],
        )
        record["process_lifetime_diagnostic_forward_ordinal"] = (
            process_lifetime_diagnostic_forward_ordinal(
                configuration.round,
                configuration.configuration_ordinal,
                repetition,
                record["pair_local_forward_ordinal"],
            )
        )
    for step in result["steps"]:
        for side in ("left", "right"):
            record = step[side]
            record.update(shared)
            record["round_relative_global_forward_ordinal"] = round_relative_global_forward_ordinal(
                configuration.configuration_ordinal,
                repetition,
                record["pair_local_forward_ordinal"],
            )
            record["process_lifetime_diagnostic_forward_ordinal"] = (
                process_lifetime_diagnostic_forward_ordinal(
                    configuration.round,
                    configuration.configuration_ordinal,
                    repetition,
                    record["pair_local_forward_ordinal"],
                )
            )


def _last_two_checks(bank: Any, configuration: Any) -> list[dict[str, Any]]:
    from formic.science.identity.crossover_diagnostic import (
        MEASURED_REPETITIONS,
        stability_against_prior,
    )

    prior_repetition = MEASURED_REPETITIONS - 2
    current_repetition = MEASURED_REPETITIONS - 1
    checks = []
    for side in ("left", "right"):
        for step in range(8):
            prior_meta, prior_logits = bank.get(
                configuration.calendar,
                configuration.configuration_ordinal,
                configuration.pair,
                prior_repetition,
                side,
                step,
            )
            current_meta, current_logits = bank.get(
                configuration.calendar,
                configuration.configuration_ordinal,
                configuration.pair,
                current_repetition,
                side,
                step,
            )
            stability = stability_against_prior(
                current_meta, current_logits, prior_meta, prior_logits
            )
            checks.append(
                {
                    "side": side,
                    "step": step,
                    "endpoint": current_meta["endpoint"],
                    "pair_local_forward_ordinal": current_meta[
                        "pair_local_forward_ordinal"
                    ],
                    "round_relative_global_forward_ordinal": current_meta[
                        "round_relative_global_forward_ordinal"
                    ],
                    "exact": stability["exact"],
                    "metric": stability,
                }
            )
    return checks


def _configuration_payload(
    configuration: Any,
    repetitions: list[dict[str, Any]],
    *,
    all_stability_exact: bool,
    final_last_two: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    from formic.science.identity.crossover_diagnostic import MEASURED_REPETITIONS

    return {
        "schema_version": 1,
        "status": status,
        **configuration.to_dict(),
        "ordinal_name": "round_relative_global_forward_ordinal",
        "repetitions": repetitions,
        "all_stability_exact": all_stability_exact
        and len(repetitions) == MEASURED_REPETITIONS,
        "final_last_two_slot_checks": final_last_two,
        "all_last_two_slots_exact": bool(final_last_two)
        and all(item["exact"] for item in final_last_two),
    }


def _configuration_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "round",
            "calendar",
            "pair",
            "configuration_ordinal",
            "status",
            "all_stability_exact",
            "all_last_two_slots_exact",
        )
    }


def _check_live_execution_storage(
    model: Any,
    output: Path,
    summaries: list[dict[str, Any]],
    label: str,
) -> None:
    summary = _write_live_summary(model, output, summaries, label)
    other = summary["storage_bytes_by_category"].get("other_python_reachable", 0)
    if other:
        raise CrossoverBlocked(
            f"Python-reachable CUDA execution storage remains after {label}: {other} bytes"
        )


def _write_live_summary(
    model: Any,
    output: Path,
    summaries: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    from formic.science.identity.artifacts import atomic_write_json
    from formic.science.identity.memory import live_cuda_tensor_summary

    gc.collect()
    summary = {"label": label, **live_cuda_tensor_summary(model)}
    summaries.append(summary)
    atomic_write_json(
        output / "live_tensors.json",
        {"schema_version": 1, "summaries": summaries},
    )
    return summary


def _existing_live_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing live tensor evidence is invalid") from exc
    if set(value) != {"schema_version", "summaries"} or value["schema_version"] != 1:
        raise RuntimeError("existing live tensor evidence has invalid schema")
    if not isinstance(value["summaries"], list):
        raise RuntimeError("existing live tensor summaries are invalid")
    return list(value["summaries"])


def _prevalidate_manifest(path: Path) -> None:
    import hashlib

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("--resume manifest is unreadable or invalid") from exc
    expected = {
        "schema_version",
        "identity",
        "completed_configurations",
        "completed_rounds",
    }
    if set(value) != expected or value["schema_version"] != 1:
        raise SystemExit("--resume manifest has an invalid schema")
    if not isinstance(value["identity"], dict):
        raise SystemExit("--resume manifest has an invalid identity")
    for key, directory in (
        ("completed_configurations", "configurations"),
        ("completed_rounds", "rounds"),
    ):
        checkpoints = value[key]
        if not isinstance(checkpoints, dict):
            raise SystemExit("--resume manifest has invalid checkpoint entries")
        for item_id, expected_digest in checkpoints.items():
            checkpoint = path.parent / directory / f"{item_id}.json"
            if (
                not isinstance(item_id, str)
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or not checkpoint.is_file()
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_digest
            ):
                raise SystemExit(f"--resume checkpoint is missing or corrupt: {key}/{item_id}")


if __name__ == "__main__":
    raise SystemExit(main())
