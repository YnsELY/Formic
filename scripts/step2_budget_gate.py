#!/usr/bin/env python3
"""Write and display the non-blocking SPEC-02 post-preflight duration estimate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from formic.science.identity.budget import (  # noqa: E402
    EstimateError,
    load_preflight_estimate,
    report_estimate,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "step2" / "preflight" / "estimate_report.json",
    )
    args, unsupported = parser.parse_known_args()
    try:
        if args.preflight is None:
            raise EstimateError("--preflight is required to produce an estimate")
        report = report_estimate(load_preflight_estimate(args.preflight))
    except (EstimateError, OSError, ValueError, json.JSONDecodeError) as exc:
        _atomic_json(
            args.output,
            {"schema_version": 1, "report": "ESTIMATE_ERROR", "error": str(exc)},
        )
        print(f"PREFLIGHT ESTIMATE: ERROR — {exc}")
        return 0
    if unsupported:
        print("PREFLIGHT ESTIMATE: ignored unsupported arguments: " + " ".join(unsupported))
    _atomic_json(args.output, report.to_dict())
    print("PREFLIGHT ESTIMATE:")
    print(
        f"  model processes={report.model_processes} "
        f"load={report.model_load_seconds:.1f}s ({report.model_load_seconds / 60:.2f} min)"
    )
    print(
        f"  preflight={report.preflight_elapsed_seconds:.1f}s "
        f"remaining={report.remaining_estimated_seconds:.1f}s"
    )
    for phase in report.phases:
        print(
            f"  {phase.name}: {phase.estimated_seconds:.1f}s "
            f"({phase.estimated_seconds / 60:.2f} min, {phase.forwards} forwards)"
        )
    print(
        f"  TOTAL ESTIMATED: {report.total_estimated_seconds:.1f}s "
        f"({report.total_estimated_seconds / 3_600:.3f} h, {report.total_forwards} forwards)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
