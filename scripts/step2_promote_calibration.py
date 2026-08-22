#!/usr/bin/env python3
"""Promote human-reviewed SPEC-02 candidate tolerances into ``tolerances.json``.

This command intentionally does not update an ADR, governance record, verdict,
or Git history.  It is a local post-pod preparation step after reviewing the
raw A40 artefacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--justifications", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tolerances.json")
    args = parser.parse_args()
    from formic.science.identity.promotion import promote_candidate_tolerances

    try:
        promote_candidate_tolerances(
            candidate_path=args.run_dir / "tolerances.candidate.json",
            raw_measurements_path=args.run_dir / "calibration" / "raw_measurements.json",
            justifications_path=args.justifications,
            output_path=args.output,
        )
    except Exception as exc:  # noqa: BLE001 - CLI renders a complete diagnostic
        print(f"SPEC-02 PROMOTION: FAIL — {type(exc).__name__}: {exc}")
        return 1
    print(f"SPEC-02 PROMOTION: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
