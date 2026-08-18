# ADR-0002 — Text-only backbone loading via `Qwen3_5ForCausalLM` + key renaming

- **Status:** PROPOSED (needs human sign-off before step 2)
- **Date:** 2026-08-16 (updated 2026-08-18 for SPEC-01 verification)
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

SPEC-01 exposes no active multimodal loader. Its direct Hugging Face reference is
the same stock `Qwen3_5ForCausalLM` class, loaded outside Formic's loader with
the same text config and prefix rename. This isolates Formic's inventory,
boundary, and runner integration without constructing a vision tower on either
side.

### Key-mapping proof

The rename is validated before weight loading as a complete mapping over the
intentionally loaded checkpoint tensors:

```text
source namespace: 850 model.language_model.* tensors + lm_head.weight
target namespace: 850 model.* tensors                  + lm_head.weight
result:           851 source names -> 851 unique target names
```

`CheckpointInventory.text_only_name_mapping()` fails on any target collision.
`text_only_mapping_report()` verifies all of the following independently of the
post-load comparison:

- source names equal exactly the checkpoint's text + LM-head inventory;
- target names equal exactly the expected CausalLM parameter names;
- inversion returns every original source name (bijection in both directions);
- the regex passed to Hugging Face produces exactly the same target name;
- shape, BF16 dtype, and parameter count are unchanged for every pair.

`tests/test_inventory.py::test_text_only_key_mapping_is_a_strict_metadata_preserving_bijection`
runs this proof over all 851 real checkpoint records. No tensor data is split,
merged, transposed, converted, added, or assigned a new role.

## Audit constraints engaged

- **A7** — satisfied structurally: the class never constructs a vision tower.
  `load_backbone` additionally asserts no `visual` module exists, and the step-1
  acceptance reports the parameter delta as evidence
  (26,895,998,464 text-only vs 27,356,728,560 multimodal = 460,730,096 = the
  vision tower, exactly the audited figure).
- **A10** — MTP is never loaded; its 15 tensors are an explicit
  exclusion, verified by count, not silently ignored.
- **A11** — no cell is re-implemented, subclassed or patched. The runtime and
  direct reference both use the stock HF CausalLM class. The renaming touches
  key *strings*, never tensors.
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
  strict expectation set. The full bijection test prevents either side from
  silently drifting.
- SPEC-01 compares Formic against a direct stock CausalLM run. Prefill logits are
  bit-identical on all six frozen prompts (maximum logit delta 0, KL 0, top-1
  agreement 6/6).
- Cached generation is not reproducible on the audited CUDA fallback backend:
  PyTorch reports that `cumsum_cuda_kernel`, used by the stock GDN prefill while
  constructing recurrent cache state, has no deterministic implementation.
  Formic does not patch or replace that cell. The preliminary generation
  criterion therefore remains failed pending a human decision; this ADR remains
  PROPOSED.
- SPEC-02, not this ADR or SPEC-01, owns measured numeric tolerances and the
  blocking identity CI gate.

## Evidence

- SPEC-01 acceptance artefacts: `artifacts/step1/step1_report.md`,
  `formic_outputs.json`, `hooks_outputs.json`, `hf_outputs.json`.
- Mapping proof: `tests/test_inventory.py` and the `key_mapping` section of the
  strict-load report.
- Audit sources: `02_config_architecture.md` (`language_model_only` inoperative),
  `10_vision_audit.md` (fusion contract), `09_mtp_audit.md` (MTP ignored by the
  runtime), `15_architectural_invariants.md` (loading invariants).
