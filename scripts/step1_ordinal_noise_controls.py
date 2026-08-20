#!/usr/bin/env python3
"""Same-path noise-floor and execution-ordinal controls for SPEC-01.

This diagnostic imports the already gated top-level observer implementation
from ``step1_runner_state_diagnostics``. It changes no model, cache, kernel,
configuration, status, or ADR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1" / "ordinal_noise_controls"
RUNNER_DIAGNOSTIC_DIR = REPO_ROOT / "artifacts" / "step1" / "runner_state_diagnostics"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
INSTANCE_NAMES = ("runner_a", "runner_b", "explicit_a", "explicit_b")
BASE_PATH = {
    "runner_a": "formic_runner",
    "runner_b": "formic_runner",
    "explicit_a": "hf_explicit",
    "explicit_b": "hf_explicit",
}
ORDINAL_WARMUPS = (0, 3, 6)
ORDER_PHASES = ("runner-first", "explicit-first")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _diagnostics() -> Any:
    import scripts.step1_runner_state_diagnostics as diagnostics

    return diagnostics


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pip_freeze_without_repo_pythonpath() -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        (sys.executable, "-m", "pip", "freeze"),
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"pip freeze failed: {result.stderr.strip()}")
    return {
        "command": f"{sys.executable} -m pip freeze",
        "environment_note": "PYTHONPATH removed for package metadata discovery",
        "stdout": result.stdout,
        "sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }


def _write_json(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[artifact] {path.relative_to(REPO_ROOT)}", flush=True)


def _write_tensors(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    _diagnostics()._torch().save(payload, path)
    print(f"[artifact] {path.relative_to(REPO_ROOT)}", flush=True)


def _metadata(
    config: Any, handle: Any, stage: str, pip_freeze: dict[str, Any]
) -> dict[str, Any]:
    from formic.science.determinism import environment_report, git_commit, git_dirty

    diagnostics = _diagnostics()
    environment = environment_report()
    model = handle.describe()
    source_manifest = diagnostics._runtime_source_manifest(config)
    protocol_identity = {
        "config_hash": config.config_hash(),
        "prompt_set_sha256": diagnostics._prompt_set()["set_sha256"],
        "forced_token_ids": list(diagnostics.FORCED_TOKEN_IDS),
        "runtime_source_manifest_sha256": diagnostics._json_sha256(source_manifest),
        "model": {
            "model_class": model["model_class"],
            "parameters": model["parameters"],
            "dtype": model["dtype"],
            "attn_implementation": model["attn_implementation"],
        },
        "environment": {
            key: environment.get(key)
            for key in (
                "torch",
                "cuda_version",
                "gpus",
                "transformers",
                "accelerate",
                "safetensors",
                "cudnn_deterministic",
                "cudnn_benchmark",
                "cudnn_allow_tf32",
                "cuda_matmul_allow_tf32",
                "flash_sdp",
                "mem_efficient_sdp",
                "math_sdp",
            )
        },
        "pip_freeze_sha256": pip_freeze["sha256"],
    }
    metadata = {
        "stage": stage,
        "created_unix_ns": time.time_ns(),
        "config_hash": config.config_hash(),
        "prompt_set_sha256": diagnostics._prompt_set()["set_sha256"],
        "forced_token_ids": list(diagnostics.FORCED_TOKEN_IDS),
        "git_commit": git_commit(REPO_ROOT),
        "git_dirty": git_dirty(REPO_ROOT),
        "environment": environment,
        "pip_freeze": pip_freeze,
        "model": model,
        "runtime_source_manifest": source_manifest,
        "protocol_identity": protocol_identity,
        "protocol_identity_sha256": diagnostics._json_sha256(protocol_identity),
    }
    metadata["control_source_sha256"] = _source_sha256()
    return metadata


def _validated_handle(
    config_path: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    diagnostics = _diagnostics()
    observer_path = RUNNER_DIAGNOSTIC_DIR / "observer_gate.json"
    if not observer_path.is_file():
        raise RuntimeError("the blocking observer gate artifact is missing")
    observer_gate = json.loads(observer_path.read_text(encoding="utf-8"))
    if not observer_gate.get("gate", {}).get("passed"):
        raise RuntimeError("the blocking observer gate did not pass")
    # Capture package state before Accelerate installs model offloading hooks.
    pip_freeze = _pip_freeze_without_repo_pythonpath()
    config, handle = diagnostics._load_handle(config_path)
    diagnostics._validate_protocol(config)
    return config, handle, observer_gate, pip_freeze


def _assert_protocol_identity(metadata: dict[str, Any], observer_gate: dict[str, Any]) -> None:
    diagnostics = _diagnostics()
    current = dict(metadata["protocol_identity"])
    observer = dict(observer_gate["metadata"]["protocol_identity"])
    current_pip = current.pop("pip_freeze_sha256")
    observer_pip = observer.pop("pip_freeze_sha256")
    if diagnostics._json_sha256(current) != diagnostics._json_sha256(observer):
        raise RuntimeError("observer gate is stale or runtime-incompatible")
    metadata["observer_gate_protocol_identity_sha256"] = observer_gate["metadata"][
        "protocol_identity_sha256"
    ]
    metadata["observer_gate_pip_freeze_sha256"] = observer_pip
    metadata["current_pip_freeze_sha256"] = current_pip
    metadata["pip_freeze_identity_note"] = (
        "Package versions are unchanged; Formic is now rendered by pip freeze as "
        "an editable local install instead of a version-only line."
    )


def _instance_trace_functions(
    handle: Any, input_ids: Any, attention_mask: Any
) -> dict[str, Callable[[], tuple[Any, ...]]]:
    diagnostics = _diagnostics()
    return {
        name: diagnostics._trace_function(BASE_PATH[name], handle, input_ids, attention_mask)
        for name in INSTANCE_NAMES
    }


def _rotated_instances(cycle: int) -> tuple[str, ...]:
    rotation = cycle % len(INSTANCE_NAMES)
    return INSTANCE_NAMES[rotation:] + INSTANCE_NAMES[:rotation]


def _ordinal_path_order(cycle: int, phase: str) -> tuple[str, str]:
    if phase not in ORDER_PHASES:
        raise ValueError(f"unknown order phase {phase!r}")
    runner_first = phase == "runner-first"
    if cycle % 2:
        runner_first = not runner_first
    return ("formic_runner", "hf_explicit") if runner_first else ("hf_explicit", "formic_runner")


def _comparison(
    left_trace: tuple[Any, ...],
    right_trace: tuple[Any, ...],
    left_states: list[dict[str, Any]],
    right_states: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = _diagnostics()
    logits = diagnostics._trace_metrics(left_trace, right_trace)
    state = diagnostics._state_component_diff(left_states, right_states)
    first_state_index = (
        state["first_difference"]["boundary_index"]
        if state["first_difference"] is not None
        else None
    )
    first_logit_index = logits["first_divergence"]
    return {
        "logits": logits,
        "state": state,
        "first_state_divergence_precedes_first_logit_divergence": (
            first_state_index is not None
            and first_logit_index is not None
            and first_state_index < first_logit_index
        ),
    }


def _aggregate_comparisons(prompts: dict[str, Any], comparison_name: str) -> dict[str, Any]:
    exact = 0
    top1 = 0
    steps = 0
    for prompt in prompts.values():
        metrics = prompt["comparisons"][comparison_name]["logits"]
        exact += metrics["exact_steps"]
        top1 += metrics["top1_matches"]
        steps += metrics["steps"]
    return {"exact_steps": exact, "top1_matches": top1, "steps": steps}


def stage_same_path_floor(config_path: Path) -> None:
    """Compare two realizations of each path under one shared schedule."""
    from formic.science.determinism import configure_determinism

    diagnostics = _diagnostics()
    torch = diagnostics._torch()
    config, handle, observer_gate, pip_freeze = _validated_handle(config_path)
    metadata = _metadata(config, handle, "same_path_floor", pip_freeze)
    _assert_protocol_identity(metadata, observer_gate)
    warmups = config.numerics.warmup_traces_per_shape
    measured_count = config.numerics.measured_traces_per_shape
    result = {
        "metadata": metadata,
        "protocol": {
            "instances": list(INSTANCE_NAMES),
            "base_path": BASE_PATH,
            "modes": ["naked", "state_captured"],
            "warmups_per_instance_mode_prompt": warmups,
            "measured_per_instance_mode_prompt": measured_count,
            "instance_order_rotates": True,
            "mode_order_alternates": True,
            "fresh_cache_per_trace": True,
        },
        "prompts": {},
    }
    tensor_payload = {}
    failures = []
    for prompt in diagnostics._render_prompts(handle.tokenizer, config):
        encoded = handle.tokenizer(prompt["text"], return_tensors="pt")
        device = diagnostics._input_device(handle.model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        trace_functions = _instance_trace_functions(handle, input_ids, attention_mask)
        measured: dict[str, dict[str, list[tuple[Any, ...]]]] = {
            name: {"naked": [], "state_captured": []} for name in INSTANCE_NAMES
        }
        states: dict[str, list[list[dict[str, Any]]]] = {name: [] for name in INSTANCE_NAMES}
        execution_order = []
        cycles = warmups + measured_count
        for cycle in range(cycles):
            instances = _rotated_instances(cycle)
            modes = ("naked", "state_captured") if cycle % 2 == 0 else ("state_captured", "naked")
            phase = "warm" if cycle < warmups else "measure"
            execution_order.append(
                {"cycle": cycle, "phase": phase, "instances": list(instances), "modes": list(modes)}
            )
            for instance in instances:
                for mode in modes:
                    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
                    trace, observer = diagnostics._run_trace(
                        trace_functions[instance],
                        handle.model,
                        observed=mode == "state_captured",
                        capture_state=mode == "state_captured",
                    )
                    if phase == "measure":
                        measured[instance][mode].append(trace)
                        if observer is not None:
                            states[instance].append(observer.states)
                    print(
                        f"[same-path] {prompt['id']} {instance} {mode} {phase} {cycle + 1}/{cycles}",
                        flush=True,
                    )
        prompt_result = {
            "prompt_length": int(input_ids.shape[-1]),
            "execution_order": execution_order,
            "instances": {},
            "comparisons": {},
        }
        prompt_tensors = {}
        final_traces = {}
        final_states = {}
        for instance in INSTANCE_NAMES:
            naked = measured[instance]["naked"]
            captured = measured[instance]["state_captured"]
            naked_stability = diagnostics._trace_metrics(naked[-2], naked[-1])
            captured_stability = diagnostics._trace_metrics(captured[-2], captured[-1])
            inertness = diagnostics._trace_metrics(captured[-1], naked[-1])
            state_stability = diagnostics._state_component_diff(states[instance][-2], states[instance][-1])
            penultimate_completeness = diagnostics._validate_state_trace(
                states[instance][-2], int(input_ids.shape[-1])
            )
            final_completeness = diagnostics._validate_state_trace(
                states[instance][-1], int(input_ids.shape[-1])
            )
            passed = (
                naked_stability["exact_steps"] == len(diagnostics.FORCED_TOKEN_IDS)
                and captured_stability["exact_steps"] == len(diagnostics.FORCED_TOKEN_IDS)
                and inertness["exact_steps"] == len(diagnostics.FORCED_TOKEN_IDS)
                and state_stability["exact"]
            )
            if not passed:
                failures.append({"prompt": prompt["id"], "instance": instance})
            prompt_result["instances"][instance] = {
                "passed": passed,
                "naked_stability": naked_stability,
                "captured_stability": captured_stability,
                "captured_vs_naked": inertness,
                "captured_state_stability": state_stability,
                "penultimate_state_completeness": penultimate_completeness,
                "final_state_completeness": final_completeness,
            }
            final_traces[instance] = captured[-1]
            final_states[instance] = states[instance][-1]
            prompt_tensors[instance] = {
                "naked": torch.stack(list(naked[-1])),
                "state_captured": torch.stack(list(captured[-1])),
            }
        for comparison_name, left, right in (
            ("reference_vs_reference", "explicit_a", "explicit_b"),
            ("runner_vs_runner", "runner_a", "runner_b"),
            ("runner_vs_reference", "runner_a", "explicit_a"),
        ):
            prompt_result["comparisons"][comparison_name] = _comparison(
                final_traces[left], final_traces[right], final_states[left], final_states[right]
            )
        result["prompts"][prompt["id"]] = prompt_result
        tensor_payload[prompt["id"]] = prompt_tensors
    result["gate"] = {"passed": not failures, "failures": failures}
    result["aggregate"] = {
        name: _aggregate_comparisons(result["prompts"], name)
        for name in ("reference_vs_reference", "runner_vs_runner", "runner_vs_reference")
    }
    _write_json("same_path_floor.json", result)
    _write_tensors(
        "same_path_floor.pt",
        {"config_hash": config.config_hash(), "tensors": tensor_payload},
    )
    if failures:
        raise RuntimeError(f"same-path instrumentation gate failed: {failures}")


def stage_ordinal_case(config_path: Path, warmups: int, order_phase: str) -> None:
    """Run one fresh-process runner/reference ordinal configuration."""
    from formic.science.determinism import configure_determinism

    if warmups not in ORDINAL_WARMUPS:
        raise ValueError(f"warmups must be one of {ORDINAL_WARMUPS}")
    if order_phase not in ORDER_PHASES:
        raise ValueError(f"order phase must be one of {ORDER_PHASES}")
    diagnostics = _diagnostics()
    torch = diagnostics._torch()
    config, handle, observer_gate, pip_freeze = _validated_handle(config_path)
    metadata = _metadata(config, handle, "ordinal_case", pip_freeze)
    _assert_protocol_identity(metadata, observer_gate)
    result = {
        "metadata": metadata,
        "protocol": {
            "warmups_per_path_prompt": warmups,
            "measured_per_path_prompt": 2,
            "order_phase": order_phase,
            "order_alternates": True,
            "instrumentation": "none",
            "fresh_process": True,
            "fresh_cache_per_trace": True,
        },
        "prompts": {},
    }
    tensor_payload = {}
    for prompt in diagnostics._render_prompts(handle.tokenizer, config):
        encoded = handle.tokenizer(prompt["text"], return_tensors="pt")
        device = diagnostics._input_device(handle.model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        functions = {
            name: diagnostics._trace_function(name, handle, input_ids, attention_mask)
            for name in ("formic_runner", "hf_explicit")
        }
        measured: dict[str, list[tuple[Any, ...]]] = {name: [] for name in functions}
        execution_order = []
        cycles = warmups + 2
        for cycle in range(cycles):
            order = _ordinal_path_order(cycle, order_phase)
            phase = "warm" if cycle < warmups else "measure"
            execution_order.append({"cycle": cycle, "phase": phase, "paths": list(order)})
            for path_name in order:
                configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
                trace = functions[path_name]()
                if phase == "measure":
                    measured[path_name].append(trace)
                print(
                    f"[ordinal] w={warmups} {order_phase} {prompt['id']} {path_name} "
                    f"{phase} {cycle + 1}/{cycles}",
                    flush=True,
                )
        runner_stability = diagnostics._trace_metrics(
            measured["formic_runner"][-2], measured["formic_runner"][-1]
        )
        explicit_stability = diagnostics._trace_metrics(
            measured["hf_explicit"][-2], measured["hf_explicit"][-1]
        )
        comparison = diagnostics._trace_metrics(
            measured["formic_runner"][-1], measured["hf_explicit"][-1]
        )
        result["prompts"][prompt["id"]] = {
            "prompt_length": int(input_ids.shape[-1]),
            "execution_order": execution_order,
            "runner_stability": runner_stability,
            "reference_stability": explicit_stability,
            "runner_vs_reference": comparison,
        }
        tensor_payload[prompt["id"]] = {
            "runner": torch.stack(list(measured["formic_runner"][-1])),
            "reference": torch.stack(list(measured["hf_explicit"][-1])),
        }
    result["aggregate"] = {
        "exact_steps": sum(
            prompt["runner_vs_reference"]["exact_steps"] for prompt in result["prompts"].values()
        ),
        "top1_matches": sum(
            prompt["runner_vs_reference"]["top1_matches"] for prompt in result["prompts"].values()
        ),
        "steps": sum(
            prompt["runner_vs_reference"]["steps"] for prompt in result["prompts"].values()
        ),
        "runner_stable_prompts": sum(
            prompt["runner_stability"]["exact_steps"] == len(diagnostics.FORCED_TOKEN_IDS)
            for prompt in result["prompts"].values()
        ),
        "reference_stable_prompts": sum(
            prompt["reference_stability"]["exact_steps"] == len(diagnostics.FORCED_TOKEN_IDS)
            for prompt in result["prompts"].values()
        ),
    }
    stem = f"ordinal_w{warmups}_{order_phase.replace('-', '_')}"
    _write_json(f"{stem}.json", result)
    _write_tensors(
        f"{stem}.pt",
        {"config_hash": config.config_hash(), "tensors": tensor_payload},
    )


def stage_ordinal_grid(config_path: Path) -> None:
    """Spawn every ordinal case in a fresh process."""
    for warmups in ORDINAL_WARMUPS:
        for phase in ORDER_PHASES:
            command = (
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--stage",
                "ordinal-case",
                "--config",
                str(config_path),
                "--warmups",
                str(warmups),
                "--order-phase",
                phase,
            )
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"ordinal case warmups={warmups}, phase={phase} failed with {result.returncode}"
                )


def _first_state_label(comparison: dict[str, Any]) -> str:
    first = comparison["state"]["first_difference"]
    if first is None:
        return "none"
    return f"{first['boundary']} / layer {first['layer']} / {first['component']}"


def stage_report() -> None:
    floor_path = ARTIFACT_DIR / "same_path_floor.json"
    if not floor_path.is_file():
        raise RuntimeError("same_path_floor.json is missing")
    floor = json.loads(floor_path.read_text(encoding="utf-8"))
    cases = []
    for warmups in ORDINAL_WARMUPS:
        for phase in ORDER_PHASES:
            path = ARTIFACT_DIR / f"ordinal_w{warmups}_{phase.replace('-', '_')}.json"
            if not path.is_file():
                raise RuntimeError(f"missing ordinal artifact: {path}")
            cases.append(json.loads(path.read_text(encoding="utf-8")))

    lines = [
        "# SPEC-01 same-path and ordinal controls",
        "",
        "Diagnostic only. No causal conclusion, status change, ADR change, or runner correction is made.",
        "SPEC-01 remains 8/9; ADR-0004 remains PROPOSED; SPEC-02 remains unstarted.",
        "",
        "## Shared-process noise floor",
        "",
        f"Instrumentation gate: **{'PASS' if floor['gate']['passed'] else 'FAIL'}**.",
        "",
        "| Prompt | Reference/reference | Runner/runner | Runner/reference | First state reference/reference | First state runner/runner | First state runner/reference |",
        "|---|---|---|---|---|---|---|",
    ]
    for prompt_id, prompt in floor["prompts"].items():
        values = []
        states = []
        for name in ("reference_vs_reference", "runner_vs_runner", "runner_vs_reference"):
            comparison = prompt["comparisons"][name]
            metric = comparison["logits"]
            values.append(
                f"{metric['exact_steps']}/{metric['steps']} exact; top-1 {metric['top1_matches']}/{metric['steps']}; first {metric['first_divergence_boundary']}"
            )
            states.append(_first_state_label(comparison))
        lines.append(
            f"| `{prompt_id}` | {values[0]} | {values[1]} | {values[2]} | "
            f"`{states[0]}` | `{states[1]}` | `{states[2]}` |"
        )
    aggregate = floor["aggregate"]
    lines += [
        "",
        "| Comparison | Exact | Top-1 |",
        "|---|---:|---:|",
    ]
    for name in ("reference_vs_reference", "runner_vs_runner", "runner_vs_reference"):
        metric = aggregate[name]
        lines.append(
            f"| `{name}` | {metric['exact_steps']}/{metric['steps']} | {metric['top1_matches']}/{metric['steps']} |"
        )

    lines += [
        "",
        "## Ordinal sensitivity",
        "",
        "Each row is a fresh process. Only warmup count and the initial phase of the alternating path order change.",
        "",
        "| Warmups | Initial order | Exact | Top-1 | Stable runner prompts | Stable reference prompts |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        protocol = case["protocol"]
        metric = case["aggregate"]
        lines.append(
            f"| {protocol['warmups_per_path_prompt']} | `{protocol['order_phase']}` | "
            f"{metric['exact_steps']}/{metric['steps']} | {metric['top1_matches']}/{metric['steps']} | "
            f"{metric['runner_stable_prompts']}/6 | {metric['reference_stable_prompts']}/6 |"
        )

    lines += [
        "",
        "## Text-only state",
        "",
        "`rope_deltas` is absent from the text-only CausalLM entrypoint in all captured same-path states. "
        "A6 therefore has no observed state to snapshot in this text-only configuration; this statement does not extend to multimodal entrypoints.",
        "",
        "## Artifacts",
        "",
        "Full per-prompt logits, top-1 values, first divergences, state hashes, protocol order, environment, config hash, and `pip freeze` are under `artifacts/step1/ordinal_noise_controls/`.",
        "",
    ]
    path = REPO_ROOT / "reports" / "step1_ordinal_noise_controls.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path.relative_to(REPO_ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-01 same-path/ordinal controls")
    parser.add_argument(
        "--stage",
        choices=("same-path-floor", "ordinal-case", "ordinal-grid", "report"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--warmups", type=int, choices=ORDINAL_WARMUPS)
    parser.add_argument("--order-phase", choices=ORDER_PHASES)
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    prepare_backend_environment(load_config(args.config).numerics)
    if args.stage == "same-path-floor":
        stage_same_path_floor(args.config)
    elif args.stage == "ordinal-case":
        if args.warmups is None or args.order_phase is None:
            parser.error("--stage ordinal-case requires --warmups and --order-phase")
        stage_ordinal_case(args.config, args.warmups, args.order_phase)
    elif args.stage == "ordinal-grid":
        stage_ordinal_grid(args.config)
    else:
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
