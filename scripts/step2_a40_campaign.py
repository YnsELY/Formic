#!/usr/bin/env python3
"""Launch the manual one-process SPEC-02 A40 calibration campaign.

Example (from the pod checkout)::

    python scripts/step2_a40_campaign.py \
      --run-id a40-2026-08-22 --sampled-continuation-seed 0

The initial run writes a review-required tolerance candidate and exits.  It
does not claim an official PASS and it does not stop the cloud pod itself.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sampled-continuation-seed", type=int, required=True)
    parser.add_argument(
        "--gpu-max-memory",
        default="35GiB",
        help="A40 placement cap validated by the balanced crossover",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from formic.config.loader import load_config
    from formic.science.determinism import (
        configure_determinism,
        prepare_backend_environment,
    )

    config = load_config(args.config)
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            max_memory={**config.backbone.max_memory, "0": args.gpu_max_memory},
        ),
    )
    prepare_backend_environment(config.numerics)
    # Apply the pinned numerical policy immediately so the environment report
    # written into run_metadata.json reflects the flags the measurements
    # actually use.  Run a40-2026-08-27-r1 recorded torch defaults because the
    # report was produced before load_backbone applied the policy; execution
    # itself was always conformant (load_backbone re-applies it).
    configure_determinism(config.run.seed, config.run.deterministic, config.numerics)
    from formic.science.identity.campaign import run_gpu_campaign

    try:
        result = run_gpu_campaign(
            config,
            run_root=REPO_ROOT / "artifacts" / "step2" / "runs" / args.run_id,
            sampled_continuation_seed=args.sampled_continuation_seed,
            resume=args.resume,
        )
    except Exception as exc:  # noqa: BLE001 - terminal command must give a verdict
        print("IDENTITY CHECK: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        return 1
    print(f"IDENTITY CHECK: {result.message}")
    verdict = (result.candidate_verdict or {}).get("verdict")
    if result.completed and verdict not in (None, "CANDIDATE_PASS"):
        # Defence in depth: the campaign itself raises on a hard FAIL, but a
        # completed run must never exit 0 with a failing candidate verdict.
        print(f"IDENTITY CHECK: candidate verdict {verdict} — treating as FAIL")
        return 1
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
