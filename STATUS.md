# Formic — status board

One line per brick. `À VENIR` / `EN COURS` / `VALIDÉE` / `GELÉE`. A brick becomes
`VALIDÉE` only when its step's exit checklist is green **and** a human has signed
off. Nothing downstream starts before that (plan rule 1).

> **Current state — SPEC-01: 9/9 PASS, human-validated. SPEC-02: EN COURS.**
> Formic and direct stock `Qwen3_5ForCausalLM` are bit-exact at prefill and at
> aligned in-process execution ordinals. The strict 851-name checkpoint mapping
> is bijective, runner/reference arguments are identical over 96 forwards, no
> state divergence precedes the logits, and boundary/state observers are
> bit-inert. `EXP-0008` also establishes a deterministic first-execution/ordinal
> effect shared by stock Hugging Face and Formic. Independent-process CUDA
> bit-exactness is therefore a documented backend limitation outside the
> identity criterion; it must not be claimed. `generate()` is excluded as an
> oracle because it follows a different call convention from the pinned explicit
> CausalLM loop. ADR-0004 is ACCEPTED by Yanis. SPEC-02 now owns the formal
> blocking identity gate, measured tolerances, and snapshot/restore primitive.

Reference documents, in order of authority: the checkpoint audit
(`/workspace/audits/qwen3_8_27b/`) → `FINAL_TARGET_ARCHITECTURE.md` (CAPE-R) →
`docs/implementation/formic_plan_implementation_initial.md` (the plan) → this repo.

## Part 1 — foundations

| # | Brick | Step | Status | Evidence |
|---|---|---|---|---|
| 1 | Repository skeleton, conventions, ADR template | 1 | VALIDÉE | `docs/conventions.md`, `docs/adr/` |
| 2 | Experiment registry (`EXP-…`) | 1 | VALIDÉE | `experiments/REGISTRY.md` |
| 3 | Run config schema (flags OFF, thinking/sampling pinned) | 1 | VALIDÉE | `formic/config/`, `configs/default.yaml` |
| 4 | Strict tensor inventory (A12) | 1 | VALIDÉE | `formic/backbone/inventory.py`, `tests/test_inventory.py` |
| 5 | Text-only backbone load, vision tower not constructed (A7) | 1 | VALIDÉE | `formic/backbone/loader.py`, ADR-0002 |
| 6 | Hybrid group view, 16×(3 GDN + 1 attention) (A11) | 1 | VALIDÉE | `formic/backbone/groups.py`, `tests/test_groups.py` |
| 7 | 17 inert boundary insertion points | 1 | VALIDÉE | `formic/backbone/boundaries.py`, `tests/test_boundaries.py` |
| 8 | Native generation through Formic (greedy + sampled) | 1 | VALIDÉE | `formic/backbone/runner.py`, `artifacts/step1/` |
| 9 | Identity baseline + numeric tolerances (E4) | 2 | EN COURS | ADR-0005 PROPOSED; corpus v2 gelé; plan A40 horizon 8 en validation |
| 10 | Snapshot / restore / fork primitive | 2 | EN COURS | `formic/state/` |
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
