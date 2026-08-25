#!/usr/bin/env python3
"""Reassess an immutable crossover run under the proposed matched-contrast gate.

This command performs no model load and no GPU forward.  It never rewrites the
source diagnostic; the reassessment is a new hash-referenced artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from formic.science.identity.artifacts import atomic_write_json
    from formic.science.identity.crossover_diagnostic import (
        balanced_design,
        build_analysis,
        validate_balanced_design,
    )

    source = args.diagnostic.resolve()
    required = {
        "manifest": source / "manifest.json",
        "analysis": source / "analysis.json",
        "same_slot": source / "same_slot_contrasts.json",
        "inversions": source / "inversion_checks.json",
        "terminal": source / "terminal.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise SystemExit("crossover evidence is incomplete: " + ", ".join(missing))
    configurations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((source / "configurations").glob("*.json"))
    ]
    prior = json.loads(required["analysis"].read_text(encoding="utf-8"))
    same_slot = json.loads(required["same_slot"].read_text(encoding="utf-8"))[
        "contrasts"
    ]
    inversions = json.loads(required["inversions"].read_text(encoding="utf-8"))[
        "checks"
    ]
    reassessed = build_analysis(
        configurations=configurations,
        same_slot_contrasts=same_slot,
        inversion_checks=inversions,
        design_validation=validate_balanced_design(balanced_design()),
    )
    payload = {
        "schema_version": 1,
        "kind": "SPEC-02 crossover readiness reassessment",
        "source_diagnostic": str(source),
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in required.items()
        },
        "prior_status": prior.get("readiness", {}).get("status"),
        "reassessment": reassessed,
        "source_evidence_immutable": True,
        "model_loaded": False,
        "gpu_forwards": 0,
        "cause_attribution": None,
    }
    atomic_write_json(args.output, payload)
    status = reassessed["readiness"]["status"]
    print(f"CROSSOVER READINESS REASSESSMENT: {status}")
    print(f"  output={args.output}")
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
