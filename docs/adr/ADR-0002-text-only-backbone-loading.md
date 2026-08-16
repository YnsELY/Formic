# ADR-0002 — Text-only backbone loading via `Qwen3_5ForCausalLM` + key renaming

- **Status:** PROPOSED (needs human sign-off before step 2)
- **Date:** 2026-08-16
- **Step:** part 1 / step 1
- **Deciders:** pending
- **Supersedes / superseded by:** —

## Context

Audit constraint **A7** requires that text-only mode *bypass the construction of
the vision tower*, and records that `language_model_only=false` in `config.json`
is **inoperative**: `Qwen3_5ForConditionalGeneration.__init__` always builds
`Qwen3_5VisionModel`. So "text-only" cannot be a config flag; it has to be a
property of the module tree.

Three facts, verified in the installed runtime
(`transformers/models/qwen3_5/modeling_qwen3_5.py`, transformers 5.8.0):

1. `Qwen3_5ForCausalLM.__init__` builds `Qwen3_5TextModel` **only** — no vision
   tower is instantiated. It also declares
   `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`.
2. `Qwen3_5Model.__init__` sets `self.language_model = Qwen3_5TextModel(...)`.
   The text sub-tree is therefore *the same class with the same layout* in both
   entry points; only the parameter prefix differs
   (`model.language_model.*` vs `model.*`).
3. On pure text input (no `pixel_values`, no `image_grid_thw`/`video_grid_thw`,
   `rope_deltas is None`), `Qwen3_5Model.forward` calls
   `compute_3d_position_ids`, which falls into its final branch — *"Can't build
   correct 3D positions. Let the model infer it"* — and passes
   `position_ids=None` to the text model. That is exactly what
   `Qwen3_5ForCausalLM.forward` does.

The checkpoint stores text weights under `model.language_model.*` (850 tensors)
plus `lm_head.weight`, vision under `model.visual.*` (333), and MTP under `mtp.*`
(15).

## Decision

Text-only mode loads **`Qwen3_5ForCausalLM`** with:

- the checkpoint's `text_config` passed explicitly as a `Qwen3_5TextConfig`;
- `from_pretrained(..., key_mapping={r"^model\.language_model\.": "model."})`, a
  **pure prefix rename** using the transformers-native key-mapping mechanism;
- the strict inventory of `formic/backbone/inventory.py` before and after the
  load, with vision (333) and MTP (15) as *declared, counted* exclusions.

`reference_multimodal` mode keeps the stock `Qwen3_5ForConditionalGeneration`
and is used only as the reference side of comparison runs.

## Audit constraints engaged

- **A7** — satisfied structurally: the class never constructs a vision tower.
  `load_backbone` additionally asserts no `visual` module exists, and the step-1
  acceptance reports the parameter delta as evidence
  (26,895,998,464 text-only vs 27,356,728,560 multimodal = 460,730,096 = the
  vision tower, exactly the audited figure).
- **A10** — MTP is never loaded in either mode; its 15 tensors are an explicit
  exclusion, verified by count, not silently ignored.
- **A11** — no cell is re-implemented, subclassed or patched; both entry points
  are stock HF classes. The renaming touches key *strings*, never tensors: no
  split, concat, transpose or re-roling.
- **A12** — permissive loading is impossible: `assert_strict_load` compares the
  loaded parameter set against the checkpoint inventory both ways (missing /
  unexpected / shape / dtype) and fails fatally on any divergence. It also
  asserts untied embeddings (`tie_word_embeddings=false`).

## Alternatives considered

| Option | Why not |
|---|---|
| `Qwen3_5ForConditionalGeneration` and accept the tower | Violates A7 ("bypass the *construction*"), wastes 460.7M params of memory, and leaves a multimodal code path live in a text-only part 1. |
| Monkeypatch `Qwen3_5VisionModel.__init__` to a stub | Patches the HF implementation — precisely what A11 forbids, and it would be invisible to the strict-load report. |
| Build a Formic wrapper around `Qwen3_5TextModel` + `lm_head` | Re-implements `generate()` plumbing (cache handling, position inference), i.e. new bug surface in exactly the machinery the audit warns about (A1–A3). |
| Rewrite the safetensors index with renamed keys in a shadow directory | Names live inside the shard files, not only in the index; HF would still look up the original names. |

## Consequences

- Text-only is a structural property, checkable in one line
  (`vision_tower_present == False`), not a promise.
- The `key_mapping` is Formic's only checkpoint-name transformation, and it lives
  in one place (`CheckpointInventory.key_mapping`) shared by the loader and the
  expectation set, so the two can never drift apart.
- Step 2 must prove the equivalence *empirically*, not only by code reading: the
  identity suite compares Formic text-only against stock
  `Qwen3_5ForConditionalGeneration` on the frozen prompt set (per-layer hidden
  states, logits, GDN/KV states, greedy identity). The step-1 acceptance already
  runs the preliminary version of that comparison.
- If step 2 ever finds a divergence attributable to the entry point, this ADR is
  superseded and text-only falls back to the multimodal class with the tower
  loaded but unused — at the cost of A7.

## Evidence

- Step-1 acceptance artefacts: `artifacts/step1/step1_report.md`,
  `formic_outputs.json`, `hf_outputs.json`.
- Audit sources: `02_config_architecture.md` (`language_model_only` inoperative),
  `10_vision_audit.md` (fusion contract), `09_mtp_audit.md` (MTP ignored by the
  runtime), `15_architectural_invariants.md` (loading invariants).
