#!/usr/bin/env bash
# Fast CI gate: everything that can be checked without loading the 55 GB checkpoint.
#
# Step 2 adds the blocking IDENTITY CHECK to this gate; from then on, any break
# freezes development until it is fixed (plan, step 2).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

echo "== formic verify (structure + inventory + config, no weights) =="
python3 -m formic.cli verify

echo
echo "== SPEC-02 identity mechanics (stock toy model, no weights) =="
python3 -m formic.cli identity-check --toy

echo
echo "== weight-free test suite =="
python3 -m pytest tests/ -q

echo
echo "== audit-constraint guards (A11 / A12 / inert-by-default) =="
python3 -m pytest \
  tests/test_no_cell_reimplementation.py \
  tests/test_inventory.py \
  tests/test_boundaries.py -q

echo
echo "CI FAST GATE: PASS"
