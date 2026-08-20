# Formic — status board

One line per brick. `À VENIR` / `EN COURS` / `VALIDÉE` / `GELÉE`. A brick becomes
`VALIDÉE` only when its step's exit checklist is green **and** a human has signed
off. Nothing downstream starts before that (plan rule 1).

> **Current state — SPEC-01 preliminary verification: 8/9 PASS, 1 FAIL — blocking.**
> Formic and a direct stock `Qwen3_5ForCausalLM` produce bit-identical prefill
> logits on 6/6 prompts (max delta 0, KL 0, top-1 6/6). The strict 851-name
> checkpoint mapping is bijective, and all 17 registered no-op hooks preserve
> real-checkpoint logits bit-for-bit in one process. The original unaligned
> cached-generation comparison fails (manual 0/4, greedy 0/6, sampled 0/3).
> The stock GDN fallback uses a CUDA `cumsum` with no deterministic torch 2.4
> implementation, but follow-up evidence does not attribute the measured logit
> gap causally to that operation. Formic does not patch or replace the cell (A11).
> Report: [`reports/step1_report.md`](reports/step1_report.md), artefacts:
> `artifacts/step1/`. This is a preliminary verification, not an identity gate;
> measured tolerances and blocking CI belong to SPEC-02. **SPEC-02 must not start
> until SPEC-01 is human-validated.** Clean run: `EXP-0007` at implementation
> commit `4e99e9a`. Follow-up `EXP-0008` establishes exact aligned CUDA
> Formic/HF decode (8/8), exact CPU Formic/HF decode (3/3), and a deterministic
> execution-ordinal effect shared by Formic and stock HF. Run 2 = run 3 exactly,
> three one-trace processes are mutually
> exact, no parameter/buffer/module tensor attribute changes, and three measured
> post-warmup traces are exact. The candidate `N=6` shape-specific warmup protocol
> makes each process stable (6/6 prompts), but a rerun with the pinned cuBLAS and
> SDPA settings still has Formic/HF generation exact on 0/4 manual, 0/6 greedy,
> and 0/3 sampled prompts across separate processes. This is a deterministic
> backend effect, not a demonstrated wrapper mismatch or random-noise floor.
> ADR-0004 is PROPOSED; no statistical criterion or SPEC-02 tolerance is approved.
> ADR-0002 remains PROPOSED.

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
