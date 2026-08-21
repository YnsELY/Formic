# Formic conventions

Working rules for anyone (human or agent) touching this repository. They exist
so that a number produced today is still interpretable in six months.

## 1. Order of authority

When two documents disagree:

```text
checkpoint audit  >  FINAL_TARGET_ARCHITECTURE.md (CAPE-R)  >  implementation plan  >  this repo
```

The audit describes what the weights *physically are*. Nothing in Formic may
assume a behaviour the audit does not confirm.

## 2. Non-negotiable rules (from the plan)

1. **Strict step order**, 1 → 8. A step starts only when the previous exit
   checklist is fully green and a human has validated it.
2. **The agent never decides architecture.** No unspecified mechanism, no
   modified threshold, no "equivalent" shortcut. Ambiguity is raised as a precise
   question, not resolved unilaterally.
3. **Every new behaviour lives behind a flag, OFF by default.** "All flags OFF"
   must always reproduce Qwen3.8-27B.
4. **Every step produces**: a dedicated branch, its tests, a short step report
   (done / measured / deviations), a `STATUS.md` update, plus an ADR for any
   structuring decision.
5. **Every step begins** with a detailed technical breakdown submitted for
   validation before implementation.
6. **Part-1 scope**: text only, Python-only target repositories, batch 1, one
   candidate per transaction, no training before step 8. BF16 for every decisive
   measurement; quantised formats may be used to iterate, never to close a
   checklist.

## 3. Audit constraint registry (A1–A12)

Re-read this at the start of every step; each detailed step SPEC must cite the
constraints it engages. Violating one is a *silent* bug.

| Code | Constraint |
|---|---|
| A1 | `use_cache=False` does **not** make a forward read-only. Never rely on it to protect a provided cache. |
| A2 | Never build `DynamicCache()` without the model config (wrong layer classes at GDN indices). |
| A3 | `Cache.crop()` restores attention KV **only**; no-op on GDN. Never a hybrid rollback. |
| A4 | Any restored/forked GDN buffer is **deep-cloned per consumer** (decode does in-place `copy_`). Flags, dtype, device travel with the tensors. No GDN buffer sharing. |
| A5 | Zero-centred RMSNorm `((1+weight)·RMS)` everywhere on the text side and in MTP; **exception**: the GDN gated norm `weight·RMS(x)·silu(z)`. Never "unify" the conventions. |
| A6 | `rope_deltas` is model-attached state, **outside** the cache; under the transactional model it is reset/derived per transaction. |
| A7 | Text-only = bypass the **construction** of the vision tower in code. `language_model_only` is inoperative in this runtime. |
| A8 | GDN padding at batch > 1: `apply_mask_to_padding_states` is conditional. Part 1 is batch 1, strictly. |
| A9 | K is cached post-norm/post-RoPE, V raw. Any cache-touching code respects this layout. |
| A10 | MTP: 424.7M params present but ignored by the runtime. Never load permissively. Out of scope in part 1. |
| A11 | The 16 hybrid groups are a **partition/view** over intact HF modules (attention at layers 3, 7, …, 63, 0-indexed). No cell re-implementation, no copy-modified code. |
| A12 | Weight loading: strict tensor inventory. Any missing or unexpected original tensor is **FATAL**. Permissive loading forbidden. |

Automated guards: A11 → `tests/test_no_cell_reimplementation.py`;
A12 → `formic/backbone/inventory.py` + `tests/test_inventory.py`;
"inert by default" → `tests/test_boundaries.py`.

## 4. Frozen cross-cutting policies

- **Thinking (plan 2.1).** The native `<think>` segment is allowed before any
  typed action, hard cap (default 4096). Scratch content is non-authoritative:
  never parsed as an action, never satisfies a criterion, never persisted; logged
  for audit only. The thinking configuration (on / off / capped-N) is pinned and
  recorded in every run.
- **Sampling (plan 2.2).** Control fields (action type, target IDs, paths,
  hashes, statuses, state transitions): **greedy, always**. Payload: checkpoint
  defaults (T=1.0, top-p=0.95, top-k=20) until the step-3 sweep decides. Scratch
  follows payload settings, inside the cap.
- **TNPR (plan 2.3).** Every transaction packet is a valid instance of the native
  chat/tool template: system ← ContractIR, user ← active obligation + criteria,
  tool turns ← evidence, assistant ← where decoding starts. Canonical content
  order (already applied, prepares part-2 HSPC): contract → stable repository
  metadata → pinned evidence sorted by stable keys → volatile tail.
  **No new token exists in steps 1–7**; the 243 reserved rows are introduced in
  step 8 together with their training.
- **Numerics (plan 2.4).** Bit-exactness is required only at identical config and
  backend. Cross-configuration comparisons (segmented vs monolithic, cached vs
  uncached, restore-continuation) use the tolerances measured in step 2
  (`tolerances.json`, a versioned artefact). Reproducibility = same kernels, same
  batch layout, same seed.

## 5. Reproducibility contract

Any quoted number carries: config hash, git commit, applicable seeds,
environment report, `EXP-…` id. SPEC-02 forced-continuation inference uses
≥3 repetitions and no RNG seed sweep; three seeds apply only when the reference
sampled continuations are generated. Any later stochastic decisive measurement
uses ≥3 seeds and reports inter-seed noise.

```bash
python -m formic.cli config    # resolved config + hash
python -m formic.cli env       # backend/environment record
python -m formic.cli verify    # weight-free structural verification (CI)
```

## 6. Naming

Branches `step-<n>-<slug>`; experiments `EXP-NNNN`; ADRs
`ADR-NNNN-<slug>.md`; artefacts `artifacts/step<N>/…`; step reports
`reports/step<N>_report.md`. Full table in ADR-0001.

## 7. Code rules

- Wrap, never rewrite: Formic composes stock Hugging Face modules.
- New behaviour = new flag in `FlagsSection`, defaulting to `False`, refused by
  the schema if it belongs to part 2.
- Config errors are fatal; unknown YAML keys are fatal (a typo must never
  silently disable a mechanism).
- Anything that reads or writes the hybrid cache states A1–A4 in its docstring
  and is covered by a test.
