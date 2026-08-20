# ADR-0004 — Deterministic cached-decode warmup protocol

- **Status:** PROPOSED
- **Date:** 2026-08-18
- **Step:** part 1 / step 1
- **Deciders:** pending; candidate acceptance rerun remains red
- **Supersedes / superseded by:** —

## Context

SPEC-01 proves wrapper equivalence: Formic and direct stock
`Qwen3_5ForCausalLM` are bit-identical at prefill on 6/6 prompts and at cached
CUDA decode when compared at the same execution ordinal (8/8 steps for run 1
and 8/8 for run 2). CPU Formic/HF cached decode is also exact on 3/3 steps.

Unwarmed CUDA traces differ between the first and later trace in a process.
`EXP-0008` shows that this is deterministic, not a Formic mismatch or a random
noise floor:

- run 2 equals run 3 exactly (8/8, delta 0, KL 0);
- three independent one-trace processes are mutually exact and equal run 1 of
  the three-trace process;
- the state fingerprint remains unchanged across traces: all 851 parameters
  (including offloaded weights), 2 registered buffers, every direct module
  tensor attribute, and 51 public `None` state slots such as `rope_deltas`;
- Formic/HF aligned ordinals are exact in separate processes.

The effect is therefore process-global backend/runtime initialization outside
the fingerprinted model tensor state. torch 2.4 reports no deterministic CUDA
implementation for a stock GDN `cumsum`, but this fact does not identify that
operation as the cause of the measured first-use behavior.

Kernel choice is shape-sensitive. A short four-token prompt warmed with the
initial `N=1` policy did not make the 20-token prompt stable: its first-use
comparison reached delta 23.71875 and KL 18.4407962 nats. With six warmups,
the short and long shapes both pass the last-two-traces exact assertion.

The candidate configuration hash is
`ac4b4adfaa98d5454d57853ddd2d51f419cab56d9aabbafb5505bc9994f44634`.
It pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disables cuDNN and CUDA-matmul TF32,
disables Flash and memory-efficient SDPA, retains math SDPA, and requires six
unmeasured traces plus two exact measured traces for every prompt/cache shape.

## Proposed decision

Retain bit-for-bit equality as the cached-decode acceptance criterion. The
candidate warmup policy is insufficient by itself: it stabilizes individual
processes but does not establish cross-process Formic/HF equality.

Every quoted cached-decode measurement must use the resolved `numerics` policy:

1. apply the pinned CUDA backend settings before model execution;
2. for every new prompt/cache shape, execute six unmeasured fresh-cache greedy
   traces;
3. execute at least two measured traces of that same shape;
4. require exact equality of the last two measured full-logit traces; otherwise
   invalidate the measurement and report the failed stability proof;
5. retain the per-step logits or metrics needed to audit the assertion.

This policy applies independently to each computational path. Cache versus full
recomputation must warm and stabilize cache and recomputation separately before
they are compared.

The full SPEC-01 candidate acceptance rerun completed under this configuration:
each stage passed its 6/6 local stability proof, but cross-process generation
remained exact on 0/4 manual-greedy, 0/6 greedy, and 0/3 sampled prompts. The
proposal therefore remains unaccepted, the ninth checklist item remains red,
and SPEC-02 does not start.

## Audit constraints engaged

- **A1/A2** — warmup traces receive fresh model-created caches; no supplied
  cache is assumed read-only and no bare `DynamicCache` is constructed.
- **A3/A4** — no crop, rollback, restore, fork, or sharing of GDN state occurs.
- **A6** — the fingerprint explicitly captures ordinary module attributes and
  `None` slots so tensor-attached state such as `rope_deltas` cannot escape it.
- **A8** — all measurements are text-only, batch 1, and unpadded.
- **A11** — no model cell, kernel, checkpoint, or installed package is patched.
- **A12** — all Formic diagnostic loads retain the strict 851/851 inventory.

## Alternatives considered

| Option | Why not |
|---|---|
| Run one trace with no warmup | Deterministically depends on process-first-use state and is invalid. |
| Warm once globally | Rejected by the long-form probe under `N=1`; shape-sensitive initialization can remain. |
| Statistical equivalence (delta, KL, top-1) | Rejected: exact cached decode is attainable under the pinned warmup protocol. These metrics remain diagnostics, not acceptance tolerances. |
| Patch or replace the GDN fallback | Violates A11 and changes the audited backend. |
| Upgrade torch | Changes the numerical baseline and was explicitly excluded from this decision. |
| Accept top-1 agreement alone | Discards full-distribution differences that the exact protocol detects. |

## Consequences

- The numerical backend and warmup policy are part of the versioned config hash,
  environment report, and every acceptance artifact.
- A newly introduced prompt length, cache length, or path must be warmed and
  pass the exact stability assertion before its output is quoted.
- CUDA cache versus full recomputation, measured after independent warmups,
  has delta max 15.09375, KL max 5.2464554 nats, and top-1 3/8. Each individual
  path is stable 8/8 exactly; the difference is a path property, not a warmup
  artifact.
- SPEC-01 remains 8/9 and SPEC-02 remains unstarted; the candidate rerun did
  not close cross-process equality.

## Evidence

- Experiment: `EXP-0008`.
- Primary diagnostic report: `reports/step1_decode_diagnostics.md`.
- Shape probe: `artifacts/step1/decode_diagnostics/cuda_formic_shape.json`.
- Hot cache/recompute probe:
  `artifacts/step1/decode_diagnostics/cuda_formic_hot_cache_recompute.json`.
- Reproducer: `scripts/step1_decode_diagnostics.py`.
- Runtime audit: `audits/qwen3_8_27b/06_gated_deltanet_audit.md`.
