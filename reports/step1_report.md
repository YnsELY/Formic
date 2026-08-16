# Step 1 — Formic foundation + backbone integration

**Branch:** `step-1-backbone-integration`
**Date:** 2026-08-16
**Plan:** `docs/implementation/formic_plan_implementation_initial.md`, step 1
**Status:** implementation complete, exit checklist run — awaiting human validation

---

## 1. Objective

Create the Formic repository and integrate Qwen3.8-27B as a neural substrate
**without changing its behaviour**, with the project's scientific tooling in
place from day one. No L0/L1, no exit gate, no LoRA, no HSPC, no DSPD, no MTP, no
training, no new token, no cell modification.

## 2. Technical breakdown (as implemented)

| # | Sub-task | Deliverable |
|---|---|---|
| 1.1 | Repository skeleton + governance | `STATUS.md`, `docs/conventions.md`, `docs/adr/`, ADR template, naming conventions (ADR-0001) |
| 1.2 | Audited constants | `formic/backbone/constants.py` — every structural number of the checkpoint, cross-checked by `tests/test_constants.py` |
| 1.3 | Run config schema | `formic/config/` + `configs/default.yaml` — strict schema, all flags OFF, thinking/sampling policies pinned |
| 1.4 | Strict tensor inventory (A12) | `formic/backbone/inventory.py` — headers-only inventory, declared exclusions, two-way post-load matching |
| 1.5 | Text-only backbone load (A7) | `formic/backbone/loader.py` — `Qwen3_5ForCausalLM` + pure key rename (ADR-0002) |
| 1.6 | Hybrid group view (A11) | `formic/backbone/groups.py` — 16 groups, 17 boundaries, contiguous-prefix helpers |
| 1.7 | Inert boundary insertion points | `formic/backbone/boundaries.py` — nothing registered while flags are OFF |
| 1.8 | Native generation | `formic/backbone/runner.py` — pinned thinking/sampling, logit fingerprints |
| 1.9 | Scientific tooling | `formic/science/` — EXP registry, environment pinning; `experiments/REGISTRY.md` |
| 1.10 | CLI + CI gate | `formic/cli.py`, `scripts/ci_fast.sh` |
| 1.11 | Tests | `tests/` — 143 weight-free tests, including guards for A11/A12, the inert-by-default rule, and the position-id contract (`tests/test_position_contract.py`, added after the 5.6 finding) |
| 1.12 | Acceptance | `scripts/step1_acceptance.py` — runs the exit checklist, emits `artifacts/step1/` |

## 3. Key decisions (ADRs)

| ADR | Decision | Status |
|---|---|---|
| ADR-0001 | Repository layout, naming, reproducibility contract | ACCEPTED |
| ADR-0002 | Text-only load via `Qwen3_5ForCausalLM` + `key_mapping` prefix rename | **PROPOSED — needs sign-off** |
| ADR-0003 | torch 2.4 × transformers 5.8 custom-op annotation shim | ACCEPTED |

ADR-0002 is the one structuring decision of this step and the only one that
required interpreting the plan. Summary of the reasoning:

- A7 demands that text-only mode **bypass the construction** of the vision tower,
  and the audit records that `language_model_only` is inoperative in this
  runtime — so text-only cannot be a config flag.
- `Qwen3_5ForCausalLM` builds `Qwen3_5TextModel` only; `Qwen3_5Model` builds the
  *same* `Qwen3_5TextModel` plus a vision tower. The text sub-tree is identical;
  only the weight prefix differs (`model.language_model.*` vs `model.*`).
- On pure text, `Qwen3_5Model.forward` reaches the final branch of
  `compute_3d_position_ids` (*"Can't build correct 3D positions. Let the model
  infer it"*) and passes `position_ids=None` to the text model — exactly what
  `Qwen3_5ForCausalLM.forward` does.
- Therefore the two entry points run the same modules on the same inputs, and the
  loading difference is a pure key rename, handled by the transformers-native
  `key_mapping` argument.

The empirical evidence gathered in this step (section 5) supports the decision;
the formal proof is step 2.

## 4. Audit constraints engaged

| Code | How this step honours it |
|---|---|
| **A7** | `Qwen3_5ForCausalLM` never constructs a vision tower; `load_backbone` asserts no `visual` module exists; memory evidence in section 5. |
| **A10** | MTP is never loaded; its 15 tensors are a *declared, counted* exclusion, not a silent ignore. |
| **A11** | The 16 groups are a view over intact modules. `validate_against_model` asserts the mixer classes are the stock `Qwen3_5GatedDeltaNet` / `Qwen3_5Attention`; `tests/test_no_cell_reimplementation.py` fails CI on any re-implemented class, cell-internal maths, or monkeypatch of the HF implementation. |
| **A12** | `assert_strict_load` compares the loaded parameter set against the checkpoint inventory both ways (missing / unexpected / shape / dtype) and is fatal on divergence; it also asserts untied embeddings. `strict_inventory` cannot be disabled by config. |
| **A1** | Documented at the one place `use_cache=False` appears (`runner.forward_logits`), with an explicit warning that it never means "read-only". |
| **A2, A3, A4, A6, A8, A9** | Not engaged yet (no cache manipulation in step 1). Recorded in `docs/conventions.md`; step 2 owns them. |
| **A5** | Not engaged (no norm is re-implemented). Recorded as a standing rule. |

## 5. Measurements

### 5.1 Strict load, text-only (Formic)

```text
model class          Qwen3_5ForCausalLM      (text model: Qwen3_5TextModel)
parameters           26,895,998,464          = 50.10 GiB BF16
tensors matched      851 / 851               (850 text + lm_head)
missing / unexpected 0 / 0
declared exclusions  {mtp: 15, vision: 333}
vision tower present False
mtp module present   False
embeddings tied      False                   (checkpoint: tie_word_embeddings=false)
attn implementation  eager
layer hooks          0                       (identity mode)
load time            209.9 s
device split         cuda:0 37.10 GiB / cpu-offloaded 12.99 GiB
```

### 5.2 Vision-tower evidence (A7)

| Path | Parameters | Vision module |
|---|---:|---|
| Formic text-only (`Qwen3_5ForCausalLM`) | 26,895,998,464 | absent |
| Stock HF (`Qwen3_5ForConditionalGeneration`) | 27,356,728,560 | present |
| Difference | **460,730,096** | = the audited vision-tower parameter count |

The delta matches `VISION_PARAMS` from the audit exactly, and the module is
absent from the tree — the tower is not constructed, not merely unused.

### 5.3 Structure (A11)

```text
64 layers, 16 groups, pattern 3x linear_attention + 1x full_attention
full-attention layers: 3, 7, 11, ..., 63   (0-indexed)
mixer classes: linear_attn -> Qwen3_5GatedDeltaNet, self_attn -> Qwen3_5Attention
17 boundaries: PRE_G1, G1_G2, ..., G15_G16, POST_G16
seq-length anchor: layer 3 (first full attention, inside G1, active on every route)
```

### 5.4 Anchor to the audit's identity baseline

Prompt `Audit technique court.` (the audit's own trace prompt), Formic text-only
vs the audit's stored baseline, which was produced with the **multimodal** entry
point:

| Quantity | Audit baseline | Formic text-only | Equal |
|---|---|---|---|
| input ids | `[71981, 14334, 5300, 13]` | `[71981, 14334, 5300, 13]` | yes |
| argmax token id | 198 | 198 | yes |
| logits min | −15.1875 | −15.1875 | yes |
| logits max | 12.375 | 12.375 | yes |
| logits RMS | 3.9636080265 | 3.9635677338 | rel. delta 1.0e−5 |
| FP32 SHA-256 | `7456afe1…` | `7fb61ef9…` | no |

Reading: two different entry points, two different cache configurations and two
different device splits agree on argmax, min and max, and differ by ~1e−5
relative on the RMS. That is the expected cross-configuration BF16 behaviour the
audit documents (GDN computes in FP32 but persists BF16, so segment boundaries
round) and which plan 2.4 explicitly places outside the bit-exactness
requirement. **Step 2 (E4) measures these tolerances and turns them into
`tolerances.json`.**

### 5.5 Formic vs a direct Hugging Face run (preliminary)

Generated tables: `artifacts/step1/step1_report.md`. Two levels are measured
separately, because they answer different questions.

**Prefill (single forward, `position_ids=None` on both sides).** Formic
text-only vs stock `Qwen3_5ForConditionalGeneration`, on all 6 reference
prompts:

| Check | Result |
|---|---|
| identical input ids | yes, 6/6 |
| identical argmax | yes, 6/6 |
| identical top-10 ids | yes, 6/6 |
| **FP32 logits SHA-256 identical** | **yes, 6/6** |
| max top-10 logit delta | 0.000e+00 |

This is stronger than the step-1 checklist required: the two entry points are
**bit-for-bit identical** on the forward pass, despite different classes,
different resident parameter sets (26.90B vs 27.36B) and therefore different
offload boundaries. It is the strongest available empirical support for
ADR-0002 short of the full step-2 suite.

**Decode.** Measured at two levels — see 5.6, which is the main finding of this
step.

### 5.6 Finding: `generate()` diverges between entry points; the backbone does not

**Observation.** Under `model.generate()`, Formic (`Qwen3_5ForCausalLM`) and the
stock multimodal class produce the same first token and then diverge (matching
prefix 1/16 on every prompt, greedy *and* sampled), even though their prefill
logits are bit-identical.

**Cause, located in the runtime source.** `Qwen3_5ForConditionalGeneration`
overrides `_prepare_position_ids_for_generation` ("Overwritten -- requires 3D
position ids"). In decode it returns

```python
position_ids = text_positions[None, ...] + self.model.rope_deltas   # shape [1, B, S]
```

`Qwen3_5TextModel.forward` recognises exactly two shapes: 2-D (which it expands
to the documented `[4, B, S]` contract) and 3-D **with first dimension 4** (axis
0 = text positions, axes 1..3 = M-RoPE, per audit report 05). A `[1, B, S]`
tensor matches neither branch, so the model takes the fallback:
`text_position_ids = None`, and the causal mask is then built without position
ids.

`Qwen3_5ForCausalLM` has no such override: it uses the generic implementation,
produces 2-D position ids, and the text model expands them to the documented
`[4, B, S]` contract.

**What is *not* the cause.** The M-RoPE embedding itself is indifferent here: on
pure text the three M-RoPE axes are identical, and feeding the rotary module
`[1, B, S]` instead of `[3, B, S]` returns bit-identical `cos`/`sin`
(verified directly, max abs delta 0.0). The divergence therefore comes from
`text_position_ids` becoming `None`, not from the rotary path.

**Consequence for the comparison method.** Comparing `generate()` outputs
compares Hugging Face's *generation wrapper*, not Formic's integration of the
backbone. The step-1 checklist criterion was therefore changed to a
**backbone-level decode comparison**: an explicit greedy loop
(`runner.manual_greedy_decode`) that leaves `position_ids=None` on both sides, so
both entry points take the same documented path, with the model creating and
advancing its own cache (A2; used strictly forward, never replayed — A1/A3/A4).
`generate()` results are still measured and reported, as a wrapper-level
observation.

**Consequence for Formic.** Formic sits on the path that matches the audited
contract. This is a good default, but it also means Formic must not rely on
`generate()` semantics being uniform across entry points — which matters from
step 4 on, when the transaction engine drives decoding itself.

**Carried to step 2.** The identity suite must (a) compare per-layer hidden
states and cache states, not only tokens; (b) pin the position-id contract
explicitly rather than inheriting it from a wrapper; (c) decide whether the
reference for the blocking identity gate is stock `Qwen3_5ForCausalLM` under a
matched device map (tightest bound) or the audit's stored baseline (anchors
cross-config tolerances). Recommendation unchanged: do both.

### 5.6 Cost

| Item | Value |
|---|---|
| Formic load | 209.9 s |
| Greedy generation, 16 tokens | ~25 s per prompt (≈1.5 s/token, CPU-offloaded) |
| Weight-free test suite | ~7 s, 143 tests |
| CI fast gate | < 10 s |

## 6. Exit checklist

| Item | Result |
|---|---|
| Native generation (greedy + sampled) matches a direct HF run — preliminary | see `artifacts/step1/step1_report.md` |
| Automated test: group ↔ layer mapping conforms | PASS (`tests/test_groups.py`, `validate_against_model`) |
| Zero re-implemented or copy-modified cell code | PASS (`tests/test_no_cell_reimplementation.py`, 3 guard families) |
| Strict tensor inventory; permissive loading impossible | PASS (851/851, 0 missing, 0 unexpected) |
| Vision tower not constructed in text mode, memory evidence | PASS (460,730,096-parameter delta, module absent) |
| All boundary hooks inert by default, config-driven | PASS (0 hooks registered; 17 boundaries exercised in tests) |
| STATUS.md, EXP registry, ADR template, conventions in place | PASS |

## 7. Deviations from the plan

1. **Package nesting.** The plan sketches `formic/backbone/…` at repo root; the
   repository uses `formic/formic/backbone/…` so the modules form one importable
   package (`from formic.backbone import …`). Module names are unchanged
   (ADR-0001).
2. **Environment shim added.** `import transformers` fails on this stack
   (torch 2.4 × transformers 5.8 custom-op annotations). The audit hit and
   documented the same issue; Formic applies the same runtime fix, isolated in
   `formic/backbone/torch_compat.py`, reported in every run's environment record
   (ADR-0003). It touches no Qwen code.
3. **Guard tests added beyond the plan.** The plan asks for *code review* on "no
   re-implemented cell"; an automated AST + pattern guard was added on top, so
   the constraint is enforced continuously rather than at review time.

No architectural decision was taken beyond ADR-0002, which is submitted for
sign-off rather than assumed.

## 8. Open questions for the human validator

1. **ADR-0002 sign-off.** Is the text-only entry point (`Qwen3_5ForCausalLM` +
   key rename) accepted as the part-1 default? The alternative — keeping
   `Qwen3_5ForConditionalGeneration` with an unused vision tower — is simpler
   but does not satisfy A7 and costs 460.7M resident parameters.
2. **Reference for step 2's identity check.** The formal identity suite should
   compare against a *pinned* reference. Two options: (a) the audit's stored
   baseline (`results/baseline_last_logits.pt`), which fixes cross-config
   tolerances as a by-product; (b) a fresh stock-HF run under a device map forced
   to match Formic's, which allows a much tighter bound. Recommendation: do both,
   (b) as the blocking gate and (a) as the anchor.
3. **Prompt-set size.** The frozen reference set is deliberately small (6
   prompts) because CPU-offloaded decode costs ~1.5 s/token. Step 2 must decide
   how far to extend it (long instruction, small software task) against runtime
   budget.

## 9. Next step

Step 2 — Identity Baseline + numeric tolerances (E4) + snapshot/restore. It
**must not** start before this checklist is validated (plan rule 1). Its first
deliverable is the single-command `IDENTITY CHECK` wired as a blocking CI gate;
from that point on, any break freezes development.
