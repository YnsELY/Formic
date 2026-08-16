# Formic — status board

One line per brick. `À VENIR` / `EN COURS` / `VALIDÉE` / `GELÉE`. A brick becomes
`VALIDÉE` only when its step's exit checklist is green **and** a human has signed
off. Nothing downstream starts before that (plan rule 1).

> **Current state — step 1: exit checklist run, 6/7 PASS, 1 FAIL — blocking.**
> Prefill is proven bit-identical between entry points (SHA-256 match, 6/6
> prompts). Multi-step **decode** diverges heavily (logit deltas 4.8–12.6,
> starting at generation step 1–2) even in a manual greedy loop that bypasses
> `generate()` entirely and matches `position_ids=None` on both sides — this
> is NOT explained by the `generate()`-wrapper position-id issue found earlier
> (report §5.6), and NOT consistent with the ~1e-5 BF16 rounding noise the
> audit documents. Root cause is unlocated: likely the KV/GDN **cache update**
> path itself, not general numerics. Only cache *length* was compared, not
> cache *content* — that comparison is the immediate next step.
> Report: [`reports/step1_report.md`](reports/step1_report.md), artefacts:
> `artifacts/step1/`. **Step 2 must not start until this is root-caused and
> the checklist is green** (plan rule 1); ADR-0002 sign-off is blocked on it.

Reference documents, in order of authority: the checkpoint audit
(`/workspace/audits/qwen3_8_27b/`) → `FINAL_TARGET_ARCHITECTURE.md` (CAPE-R) →
`docs/implementation/formic_plan_implementation_initial.md` (the plan) → this repo.

## Part 1 — foundations

| # | Brick | Step | Status | Evidence |
|---|---|---|---|---|
| 1 | Repository skeleton, conventions, ADR template | 1 | EN COURS | `docs/conventions.md`, `docs/adr/` |
| 2 | Experiment registry (`EXP-…`) | 1 | EN COURS | `experiments/REGISTRY.md` |
| 3 | Run config schema (flags OFF, thinking/sampling pinned) | 1 | EN COURS | `formic/config/`, `configs/default.yaml` |
| 4 | Strict tensor inventory (A12) | 1 | EN COURS | `formic/backbone/inventory.py`, `tests/test_inventory.py` |
| 5 | Text-only backbone load, vision tower not constructed (A7) | 1 | EN COURS | `formic/backbone/loader.py`, ADR-0002 |
| 6 | Hybrid group view, 16×(3 GDN + 1 attention) (A11) | 1 | EN COURS | `formic/backbone/groups.py`, `tests/test_groups.py` |
| 7 | 17 inert boundary insertion points | 1 | EN COURS | `formic/backbone/boundaries.py`, `tests/test_boundaries.py` |
| 8 | Native generation through Formic (greedy + sampled) | 1 | EN COURS | `formic/backbone/runner.py`, `artifacts/step1/` |
| 9 | Identity baseline + numeric tolerances (E4) | 2 | À VENIR | — |
| 10 | Snapshot / restore / fork primitive | 2 | À VENIR | — |
| 11 | Evaluation harness + pinned baselines (E1) | 3 | À VENIR | — |
| 12 | ContractIR + three-pass compiler | 4 | À VENIR | — |
| 13 | State Fabric (repo snapshot, DAG, evidence, ledger) | 4 | À VENIR | — |
| 14 | Transaction engine + reference monitor + commit | 4 | À VENIR | — |
| 15 | Typed actions (EditIR/ToolIR/…), lowering, sandbox | 5 | À VENIR | — |
| 16 | End-to-end loop + FORMIC-M0 measurement | 6 | À VENIR | — |
| 17 | Episode machine v1 (labelled data) | 7 | À VENIR | — |
| 18 | L2 neural sidecars + FORMIC-M1 | 8 | À VENIR | — |

## Part 2 — deliberately not started

`L1/L0 exit bridges`, `route-conditional Pass-LoRA`, `anytime exit gating`,
`tight scratch budgets`, `continue/act head`, `HSPC`, `DSPD`, `GDN rollback
protocol`, `MTP`, `full constrained decoding`, `long-horizon 50–500`,
`outcome/preference training`, `RL`, `advanced serving`, `vision path`.

Each has a flag in `configs/default.yaml`, all OFF, and the config schema refuses
to enable them during part 1.

## Milestones

| Milestone | Definition | Status |
|---|---|---|
| FORMIC-M0 | Full-depth transactional executor + harness-only baseline figures | À VENIR |
| FORMIC-M1 | L2 sidecars beating M0 in isolated ablation, no full-depth regression | À VENIR |

## Standing invariants

1. All flags OFF ⇒ Formic reproduces Qwen3.8-27B. Enforced by the config schema
   today, by the blocking identity CI from step 2 onward.
2. No cell is re-implemented, copied or monkeypatched (A11) — guarded by
   `tests/test_no_cell_reimplementation.py`.
3. Strict tensor inventory; permissive loading impossible (A12).
4. Batch 1, text only, Python-only target repositories, BF16 for every decisive
   measurement (plan rule 6).
