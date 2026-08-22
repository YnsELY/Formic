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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sampled-continuation-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Importing torch is intentionally deferred until the resolved numerical
    # environment has been read and pinned.
    from formic.config.loader import load_config
    from formic.science.determinism import prepare_backend_environment

    config = load_config(args.config)
    prepare_backend_environment(config.numerics)
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
    print("STOP POD BEFORE ANALYSIS")
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
