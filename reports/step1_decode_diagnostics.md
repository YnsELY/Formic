# SPEC-01 — cached-decode diagnostics

**Status:** diagnosis complete; SPEC-01 remains **8/9**, SPEC-02 not started

**Experiment:** `EXP-0008`

**Proposal under review:** `docs/adr/ADR-0004-deterministic-cached-decode-warmup.md`

## Protocol

- Stack unchanged: torch 2.4.1+cu124, transformers 5.8.0, BF16, eager attention,
  no FLA, no `causal-conv1d`, NVIDIA A40 for CUDA runs.
- CUDA config hash: `19455e2b0c639dd9c9de967a4566f743d59dc4a241944053511b63c2a97a8ef2`.
  CPU device-only override hash: `17cef0d27a17db2f069398c42c417d3600390cea54f42c232d74cc5ec64d18ee`.
- Seed 0, batch 1, no padding, prompt `Audit technique court.` (4 tokens).
- CUDA: 8 logit steps; CPU: 3 logit steps.
- Every trace receives the same forced continuation. Metrics after an argmax
  disagreement therefore still compare identical input prefixes.
- Formic and direct HF are loaded in separate processes. Each process performs
  two fresh-cache traces. Formic additionally compares cache with full
  `use_cache=False` recomputation in the same loaded model.
- No notebook cell, Qwen cell, kernel, checkpoint, default config, or torch
  version was modified.

## Established wrapper equivalence

This is the primary SPEC-01 result:

| Comparison | Exact logits | max delta | KL | top-1 |
|---|---:|---:|---:|---:|
| Prefill, six frozen prompts | 6/6 | 0 | 0 | 6/6 |
| CUDA Formic run 1 / HF run 1 | 8/8 | 0 | 0 | 8/8 |
| CUDA Formic run 2 / HF run 2 | 8/8 | 0 | 0 | 8/8 |
| CPU Formic / HF | 3/3 | 0 | 0 | 3/3 |

Formic and the direct stock-HF reference reproduce each other bit for bit across
separate processes when compared at the same execution ordinal. The wrapper,
mapping, configuration, loading, and call sequence are therefore equivalent for
the tested paths.

## First-execution follow-up

### Three traces in one process

| Comparison | Exact logits | max delta | max KL | top-1 | first difference |
|---|---:|---:|---:|---:|---:|
| Run 1 / run 2 | 1/8 | 14.15625 | 3.4712985 | 1/8 | step 1 |
| Run 2 / run 3 | 8/8 | 0 | 0 | 8/8 | none |
| Run 1 / run 3 | 1/8 | 14.15625 | 3.4712985 | 1/8 | step 1 |

Run 2 equals run 3 exactly. This proves a deterministic first-execution effect;
it does not by itself identify whether the cause is model state or backend
initialization/autotune.

### Model-state fingerprint

The model was fingerprinted before run 1 and after runs 1, 2, and 3:

| Category | Coverage | Aggregate SHA-256 stable | Changed entries |
|---|---:|---|---:|
| Parameters | 851/851, including Accelerate-offloaded values resolved from `weights_map` | yes | 0 |
| Registered buffers | 2/2 | yes | 0 |
| Direct module `__dict__` tensor attributes | all modules; 0 tensors and 51 public `None` state slots | yes | 0 |

The public `None` slots are included so a transition such as
`rope_deltas: None -> Tensor` cannot escape the scan. No parameter, registered
buffer, ordinary tensor attribute, or such state slot changed at any boundary.
Persistent tensor state attached to the model is therefore not supported by
this diagnostic. Non-model process/global backend state remains possible.

### One trace per process

Three independent Formic processes each executed exactly one trace. Every pair
is exact on 8/8 steps, with delta 0, KL 0, and top-1 8/8. Each also equals run 1
of the three-trace process exactly, while differing from its stabilized run 2 at
step 1. "First trace in a process" is thus the explanatory variable.

### Autotune settings and warmup

The initial follow-up only seeded Python, NumPy, and Torch; it set
`cudnn.deterministic=True` and kept `cudnn.benchmark=False`. The candidate
versioned policy additionally uses:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.benchmark=False
cudnn.deterministic=True
cudnn.allow_tf32=False
cuda.matmul.allow_tf32=False
flash_sdp=False
mem_efficient_sdp=False
math_sdp=True
```

After one unmeasured cached-decode warmup, measured runs 1, 2, and 3 are exact
on 8/8 steps. The warmup itself differs from measured run 1 only at step 7
(delta 5.265625, KL 0.3236326, same top-1). Warming therefore yields a viable
bit-exact measurement protocol. Because a first-use effect remains even under
these settings, the evidence points to process-global runtime/backend
initialization state, not a specific autotuner mechanism.

### Candidate acceptance rerun

The candidate policy was then applied before Torch import to separate Formic and
direct-HF acceptance processes, with six unmeasured fresh-cache traces and two
measured traces for each of the six frozen prompt/cache shapes. Both stages pass
their local last-two-traces assertion on 6/6 prompts. Prefill remains exact on
6/6 prompts. However, the independently initialized processes still produce
identical generation on 0/4 manual-greedy, 0/6 native-greedy, and 0/3 sampled
prompts. Thus the policy establishes process-local stability only; it does not
yet make cross-process CUDA decode exact.

## Repeated-run effect

| Device / comparison | Exact logits | max delta | max KL (nats) | top-1 | first tensor / top-1 difference |
|---|---:|---:|---:|---:|---:|
| CUDA Formic / Formic | 1/8 | 14.15625 | 3.4712985 | 1/8 | 1 / 1 |
| CUDA HF / HF | 1/8 | 14.15625 | 3.4712985 | 1/8 | 1 / 1 |
| CPU Formic / Formic | 3/3 | 0 | 0 | 3/3 | none / none |
| CPU HF / HF | 3/3 | 0 | 0 | 3/3 | none / none |

The repeated-run effect is present with stock HF independently of Formic. It
must not yet be classified as random CUDA noise: Formic run 1 equals HF run 1,
Formic run 2 equals HF run 2, and both repeated-run profiles match step by step.
That execution-ordinal structure instead motivates first-execution, persistent
state, and autotune diagnostics.

## CPU Formic vs HF

The short CPU cached decode is exactly equal on 3/3 steps: delta 0, KL 0, and
top-1 3/3. This confirms that the observed non-reproducibility is specific to
the CUDA backend used here, not the Formic mapping, loading, or call sequence.

## First divergence

Aligned CUDA realizations happen to match exactly: Formic run 1 versus HF run 1
is 8/8 exact, as is run 2 versus run 2. Cross-realization Formic run 1 versus HF
run 2 exposes the same execution-ordinal difference as each repeated-run control:

| Step | max delta | KL (nats) | top-1 Formic / HF | agree |
|---:|---:|---:|---:|---|
| 0 (cache prefill) | 0 | 0 | 198 / 198 | yes |
| 1 | 5.828125 | 1.1978349 | 220 / 2 | no |
| 2 | 14.15625 | 2.9680113 | 17366 / 220 | no |
| 3 | 12.875 | 2.4211623 | 107669 / 15 | no |
| 4 | 9.6489258 | 3.0430888 | 15 / 13 | no |
| 5 | 9.765625 | 2.3150517 | 17 / 220 | no |
| 6 | 11.28125 | 3.4712985 | 16 / 15 | no |
| 7 | 10.1875 | 1.0787122 | 15 / 16 | no |

The first difference is step 1, the first token after cache construction. It is
a **frank divergence**, not an argmax switch on quasi-equality: delta 5.828125,
KL 1.1978349 nats, and preference gaps between the two selected tokens of
1.4375 in one realization and 0.4375 in the other.

There is therefore no wrapper-specific first divergence: its presence depends
on which CUDA execution ordinals are paired. The stable fact is that different
ordinals first separate at step 1, while aligned Formic/HF ordinals are
bit-identical.

## Cache vs recomputation

| Device | Exact logits | max delta | max KL (nats) | top-1 | first tensor / top-1 difference |
|---|---:|---:|---:|---:|---:|
| CUDA Formic cache / recompute | 1/8 | 14.84375 | 5.5040346 | 1/8 | 1 / 1 |
| CPU Formic cache / recompute | 1/3 | 0.15625 | 2.959149e-4 | 3/3 | 1 / none |

On CPU, cache and recomputation remain close but not bit-identical after
prefill. This is expected from the audited GDN path: the recurrent state is
computed in FP32 but persisted in BF16 at each cached token boundary, whereas a
full recomputation retains FP32 state within the multi-token segment. The prior
CUDA cache/recompute number is confounded: it compares cached run 1 with a
recomputation performed after the process had crossed its first-execution
boundary. It does not isolate a causal cache/recompute difference.

## Conclusion

Wrapper equivalence is the established result: Formic/HF is exact on 8/8 aligned
CUDA steps, exact on 3/3 CPU steps, and the prefill proof remains exact on 6/6
prompts. The repeated-run difference is a deterministic first-execution effect
outside the fingerprinted model tensor state. The candidate warmup policy makes
each process stable, but does not make separate Formic/HF processes equal. It
therefore cannot close the ninth SPEC-01 item. ADR-0004 remains PROPOSED, and no
SPEC-02 tolerance or criterion change is approved.

Full per-step metrics and environment records are in
`artifacts/step1/decode_diagnostics/report.json` and
`artifacts/step1/decode_diagnostics/followup_report.json`.
