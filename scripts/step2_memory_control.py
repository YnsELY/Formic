#!/usr/bin/env python3
"""Measure SPEC-02 CUDA residency through preflight and the first inertia pair."""

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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    config = load_config(args.config)
    prepare_backend_environment(config.numerics)

    import torch

    from formic.backbone.loader import load_backbone
    from formic.science.identity.artifacts import atomic_write_json
    from formic.science.identity.campaign import _MeasurementSession
    from formic.science.identity.campaign_plan import build_campaign_plan
    from formic.science.identity.memory import IncrementalMemoryWriter
    from formic.science.identity.preflight import release_cuda_working_set, run_preflight
    from formic.science.identity.prompts import load_frozen_corpus

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    memory = IncrementalMemoryWriter(output / "cuda_memory.json")
    memory.record("before_load")
    handle = None
    try:
        handle = load_backbone(config)
        memory.record("after_load", handle.model)
        corpus = load_frozen_corpus(REPO_ROOT / config.identity.prompt_set_path)
        corpus.validate_tokenizer(handle.tokenizer)
        plan = build_campaign_plan(config, corpus)
        run_preflight(
            handle,
            plan,
            estimate_path=output / "estimate.json",
            details_path=output / "timings.json",
            memory_observer=lambda label: memory.record(label, handle.model),
        )
        release_cuda_working_set()
        session = _MeasurementSession(handle, config, memory=memory)
        first_prompt = corpus.prompts[0]
        for _ in range(config.numerics.warmup_traces_per_shape):
            session.trace_off_prefill(first_prompt)
        release_cuda_working_set()
        before = memory.record("before_first_comparison", handle.model)
        memory.write_live_summary(handle.model)
        if before["device_free_bytes"] < 512 * 1024**2:
            raise RuntimeError("insufficient CUDA headroom before first comparison")
        metric = session.trace_off_on_pair(first_prompt)
        memory.record("after_first_comparison", handle.model)
        atomic_write_json(
            output / "terminal.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "first_comparison": metric.to_dict(),
            },
        )
        return 0
    except Exception as exc:
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
            },
        )
        raise
    finally:
        del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
