#!/usr/bin/env python3
"""Run only the pinned legacy stability case and persist every repetition.

This is a diagnostic control, not the SPEC-02 calibration or an identity
verdict.  It uses the same batch-1, cached-decode path and warmup rule as the
campaign for ``legacy__audit_echo`` while avoiding a full campaign relaunch.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True, help="new diagnostic directory")
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import environment_report, git_commit, prepare_backend_environment

    config = load_config(args.config)
    prepare_backend_environment(config.numerics)

    import torch

    from formic.backbone.loader import load_backbone
    from formic.science.identity.artifacts import atomic_write_json
    from formic.science.identity.campaign import _MeasurementSession, _assert_a40_environment
    from formic.science.identity.campaign_plan import CampaignPath, timing_continuation
    from formic.science.identity.memory import IncrementalMemoryWriter
    from formic.science.identity.prompts import load_frozen_corpus
    from formic.science.identity.types import ExecutionMode, SamplingMode

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    atomic_write_json(
        output / "run_metadata.json",
        {
            "schema_version": 1,
            "kind": "SPEC-02 legacy stability diagnostic only",
            "config_sha256": config.config_hash(),
            "git_commit": git_commit(),
            "environment": environment_report(),
        },
    )
    _assert_a40_environment()
    memory = IncrementalMemoryWriter(output / "memory" / "cuda_memory.json")
    memory.record("before_load")
    handle = None
    try:
        handle = load_backbone(config)
        memory.record("after_load", handle.model)
        corpus = load_frozen_corpus(REPO_ROOT / config.identity.prompt_set_path)
        corpus.validate_tokenizer(handle.tokenizer)
        prompt = next(item for item in corpus.prompts if item.id == "audit_echo" and item.set_name == "legacy")
        session = _MeasurementSession(handle, config, memory=memory)
        path = CampaignPath(prompt, ExecutionMode.DECODE_CACHED)

        def observe(payload: dict[str, object]) -> None:
            atomic_write_json(output / "diagnostic.json", payload)

        result = session.measure_forced(
            case_id="legacy__audit_echo",
            phase="legacy_stability_probe",
            path=path,
            forced_token_ids=timing_continuation(prompt, config.identity.decode_tokens),
            repetitions=config.identity.exact_gate_repetitions,
            sampling=SamplingMode.GREEDY,
            continuation_seed=None,
            exact_required=True,
            endpoints=None,
            logits_only=False,
            decode_steps=None,
            repetition_observer=observe,
        )
        memory.record("after_probe", handle.model)
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "message": "LEGACY STABILITY PROBE COMPLETE — NOT AN IDENTITY VERDICT",
                "result": result,
                "stop_pod_before_analysis": True,
            },
        )
        print("LEGACY STABILITY PROBE: PASS")
        print("STOP POD BEFORE ANALYSIS")
        return 0
    except Exception as exc:  # noqa: BLE001 - terminal command needs a durable failure record
        if handle is not None:
            try:
                memory.record("on_failure", handle.model)
                memory.write_live_summary(handle.model)
            except Exception:
                pass
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
        print("LEGACY STABILITY PROBE: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        print("STOP POD BEFORE ANALYSIS")
        return 1
    finally:
        del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
