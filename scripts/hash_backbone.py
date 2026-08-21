#!/usr/bin/env python3
"""Compute or validate the canonical 851-tensor backbone content hash."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formic.backbone.inventory import CheckpointInventory
from formic.science.backbone_hash import (
    canonical_backbone_hash,
    load_reusable_backbone_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reuse-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.reuse_audit is None):
        parser.error("provide exactly one of --checkpoint or --reuse-audit")
    if args.reuse_audit is not None:
        result = load_reusable_backbone_hash(args.reuse_audit)
    else:
        inventory = CheckpointInventory.from_checkpoint(args.checkpoint)
        inventory.validate_against_audit()
        result = canonical_backbone_hash(inventory)
    payload = result.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
