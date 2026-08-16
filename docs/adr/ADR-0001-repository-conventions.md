# ADR-0001 — Repository layout, naming and reproducibility conventions

- **Status:** ACCEPTED
- **Date:** 2026-08-16
- **Step:** part 1 / step 1
- **Deciders:** step-1 implementation
- **Supersedes / superseded by:** —

## Context

The plan mandates a fixed module layout (`backbone/`, `runtime/`, `contracts/`,
`state/`, `actions/`, `validation/`, `eval/`, `episodes/`, `sidecars/`,
`configs/`, `docs/adr/`, `tests/`, `scripts/`, `STATUS.md`) and requires that
every step produce a branch, tests, a step report, a `STATUS.md` update and ADRs
for structuring decisions. It also requires that a run be fully described by its
config.

Two details need a decision because the plan does not fix them: how the modules
become an importable Python package, and how runs/artefacts are named.

## Decision

**Layout.** The repository root is `/workspace/formic`. The importable package is
`formic/`, containing exactly the modules the plan names:

```text
formic/                    # repo root
├── formic/                # python package
│   ├── backbone/          # checkpoint integration, group view, boundaries (step 1-2)
│   ├── runtime/           # transaction engine (step 4)
│   ├── contracts/         # ContractIR + compiler (step 4)
│   ├── state/             # State Fabric (step 4)
│   ├── actions/           # grammar, IRs, lowering (step 5)
│   ├── validation/        # reference monitor, sandbox, checks (step 4-5)
│   ├── eval/              # harness, suites, baselines (step 3)
│   ├── episodes/          # episode machine (step 7)
│   ├── sidecars/          # neural components (step 8)
│   ├── config/            # run config schema + strict loader
│   ├── science/           # experiment registry, environment pinning
│   └── cli.py
├── configs/               # run configs (YAML) and frozen prompt sets
├── docs/adr/              # architecture decision records
├── tests/                 # weight-free by default; `weights` marker for the rest
├── scripts/               # acceptance / experiment entry points
├── experiments/           # EXP registry (append-only jsonl + rendered table)
├── artifacts/             # run outputs (git-ignored)
├── reports/               # step reports (committed)
└── STATUS.md
```

Imports are `from formic.backbone import ...` — one namespace, no top-level
module named `backbone`/`state`/`actions` that could collide with the ecosystem.

**Naming.**

| Kind | Convention | Example |
|---|---|---|
| Git branch | `step-<n>-<slug>` | `step-1-backbone-integration` |
| Experiment | `EXP-NNNN` (registry-allocated) | `EXP-0001` |
| ADR | `ADR-NNNN-<slug>.md` | `ADR-0002-text-only-backbone-loading.md` |
| Run artefacts | `artifacts/step<N>/<artefact>.json` | `artifacts/step1/formic_outputs.json` |
| Step report | `reports/step<N>_report.md` | `reports/step1_report.md` |
| Config | `configs/<name>.yaml` | `configs/default.yaml` |

**Reproducibility contract.** Any number that will be quoted must come with:
config hash (`python -m formic.cli config`), git commit, seeds, environment
report (`python -m formic.cli env`), and an `EXP-…` entry. Decisive measurements
use ≥3 seeds and BF16; quantised formats may be used to iterate, never to close
an exit checklist (plan rule 6).

**Tests.** `tests/` runs without the checkpoint weights by default; anything that
needs the 55 GB load carries the `weights` marker and lives behind the acceptance
scripts. Audit constraints get automated guards where possible
(`tests/test_no_cell_reimplementation.py` for A11, `tests/test_inventory.py` for
A12, `tests/test_boundaries.py` for the inert-by-default rule).

## Audit constraints engaged

None directly; this ADR fixes process, not model behaviour. It exists so later
ADRs can reference stable paths and names.

## Alternatives considered

| Option | Why not |
|---|---|
| Flat modules at repo root (`backbone/`, `state/`, …) | Matches the plan sketch literally but produces generic top-level import names and no packaging story. |
| `src/formic/` layout | Equivalent; adds a level without benefit at this size. |

## Consequences

- The plan's module names are preserved one-to-one under a single package.
- `artifacts/` is git-ignored (run outputs, potentially large); `reports/` and
  `experiments/` are committed (they are the project's memory).
- A future packaging step only needs `pyproject.toml` to point at `formic/`.
