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
> The A40 r2 crossover measured 1,536/1,536 exact same-slot endpoint
> comparisons while raw process ordinals varied. Campaign run
> `a40-2026-08-26-r1` (launcher v2) failed at the first legacy case: the
> first-execution realisation switch extended past the capture-free warmup
> into the first two measured pair traces
> (`reports/step2_a40_run_2026-08-26_diagnostic.md`). Protocol v3 (burn-in
> after every non-empty warmup block, per-endpoint warmups, logits-only
> 64-frame probe, RR-floored snapshot adjudication) fixed it: run
> `a40-2026-08-27-r1` passed the legacy gate 6/6 with raw
> repetition-reproducibility on every slot, then failed at the noise floor on
> the mixed reference/runner pair, which oscillates with period 2 under the
> alternating calendar while RR/NN — the pairs that produce the floor — were
> stable 3/3 (`reports/step2_a40_run_2026-08-27_diagnostic.md`). The blocking
> last-two assertion now applies to RR/NN only; the mixed pair is a recorded
> non-blocking diagnostic. Run `a40-2026-08-28-r1` then completed seven
> phases — including the noise floor on all three prompts (RR floors up to
> 19.5625) and snapshot/restore on the real checkpoint — plus the whole short
> class and twelve medium cases, before failing on medium cached decode
> (`reports/step2_a40_run_2026-08-28_diagnostic.md`). It measured the last two
> structural locks: the canonical reference repeats bit-identically over four
> executions while the cached candidate does not (continuous variability in
> the late groups from step 2), and cross-path top-1 agreement is 3/8, 1/8 and
> 2/8 on stable cases while the same-path control stays exact 8/8 at zero
> delta. Protocol v4 therefore anchors tolerance-measurement stability on the
> canonical reference (candidate variability recorded as diagnostic) and
> counts cross-position top-1 flips instead of failing on them; both stay
> blocking where the protocol is aligned, and every affected row remains
> bounded/REVIEW_REQUIRED. Run `a40-2026-08-28-r2` validated those two
> criteria by measurement — the medium class completed in full — and then
> failed on the first long segmented case with a structural mismatch:
> single-frame reference prefixes each resolved as final and captured final
> state, so the two sides' model-state registries differed. Segmented
> reference prefixes now inherit the paired candidate frame's capture
> profile, which also restores ADR-0005 long-class capture rules and lowers
> the planned long-class transfer from 18.87 to 14.07 GiB
> (`reports/step2_a40_run_2026-08-28_r2_diagnostic.md`). The 9,925-forward
> calibration remains to be rerun (runbook:
> `docs/runbooks/step2_pod_campaign.md`); no tolerance or official SPEC-02
> PASS exists yet.

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
| 9 | Identity baseline + numeric tolerances (E4) | 2 | EN COURS | ADR-0005 PROPOSED; corpus v2 gelé; launcher A40 Latin-ABBA/cross-path prêt; calibration réelle en attente |
| 10 | Snapshot / restore / fork primitive | 2 | EN COURS | `formic/state/`; tests synthétiques verts; adjudication checkpoint réel intégrée à la campagne |
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
