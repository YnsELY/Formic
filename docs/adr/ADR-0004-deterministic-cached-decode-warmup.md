# ADR-0004 — Aligned in-process identity protocol for cached decode

- **Status:** ACCEPTED
- **Date:** 2026-08-21
- **Step:** part 1 / step 1
- **Deciders:** Yanis
- **Supersedes / superseded by:** —

## Context

SPEC-01 initially compared Formic and Hugging Face cached generation in
independent CUDA processes. Prefill was bit-exact, but cached decode could
diverge by roughly 14 logit units. The diagnostics had to distinguish a Formic
wrapper error from a property of the stock backend without modifying a Qwen
cell, kernel, checkpoint, or installed package.

The following observations are established by `EXP-0008` and the step-1
diagnostic reports:

1. Formic and a direct stock `Qwen3_5ForCausalLM` are exact when compared at
   aligned execution ordinals, on CUDA and CPU.
2. The first cached trace of a process and later traces may belong to two
   different, individually stable numerical realizations. The same effect is
   present with stock Hugging Face alone.
3. Three independent one-trace processes are mutually exact. Within a
   multi-trace process, runs 2 and 3 are exact while run 1 may differ.
4. Model fingerprints remain unchanged: parameters, registered buffers,
   direct module tensor attributes, and public `None` state slots do not
   explain the ordinal effect.
5. Formic and the explicit Hugging Face loop have zero recorded argument
   differences over 96 forwards. No cache or model-state divergence precedes
   the first logit divergence.
6. Boundary observers and state capture are bit-inert under their dedicated
   gates.
7. `generate()` is not the same reference computation as the explicit loop.
   It pre-creates a cache and uses explicit masks/positions and
   `logits_to_keep=1`; the explicit loop computes the full LM-head output and
   slices the final position. The two conventions may diverge from prefill
   while retaining the same logical model.

PyTorch 2.4 reports no deterministic CUDA implementation for a stock GDN
`cumsum`. This is a backend fact, not a demonstrated root cause of the observed
ordinal effect. The project records the measurements and does not attribute
causality.

## Decision

There is no unqualified single “Hugging Face reference flow.” Formic identity
uses the **explicit CausalLM decoding loop** as its pinned reference convention;
`model.generate()` is excluded as an identity oracle.

Every identity comparison is performed at an aligned protocol and execution
ordinal inside one process. The resolved numerical policy is applied before
model execution. For each exact input shape, the path receives six unmeasured
fresh-state warmup traces, followed by at least two measured traces. The last
two measured traces must be bit-exact; otherwise the measurement is invalid.
No state is captured during warmup.

The blocking identity criterion is:

- bit-exact prefill;
- exact Formic/reference execution at an aligned in-process protocol;
- zero forward-argument differences;
- no cache or model-state divergence before a logit divergence;
- proven inertness of observers, boundary hooks, and diagnostic tracing.

Bit-for-bit identity between independent CUDA processes is classified as a
documented backend limitation and is outside the identity criterion. It must
not be claimed by Formic reports.

Cross-path comparisons such as segmented versus monolithic prefill, cached
decode versus full recomputation, and continuous versus restored execution are
separate numerical-equivalence measurements. SPEC-02 owns their measured
tolerances and blocking policy.

## Audit constraints engaged

- **A1** — no supplied cache is treated as read-only because `use_cache=False`.
- **A2** — caches are model-created or constructed with the model config.
- **A3/A4** — SPEC-01 performs no crop, rollback, restore, or shared-buffer
  fork. SPEC-02 implements those operations through deep-cloned snapshots.
- **A6** — model-attached state slots, including absence of `rope_deltas` on
  the text-only CausalLM entrypoint, are recorded explicitly.
- **A8** — all decisive runs are text-only, batch 1, and unpadded.
- **A11** — no Qwen cell or kernel is replaced, copied, subclassed, or patched.
- **A12** — all Formic loads retain the strict 851/851 textual inventory.

## Alternatives considered

| Option | Why not |
|---|---|
| Compare independent CUDA processes bit-for-bit | The stock backend itself exhibits a deterministic execution-ordinal effect, so this mixes wrapper identity with process history. |
| Use `generate()` as the Hugging Face oracle | It follows a different call and LM-head convention from the explicit loop. |
| Accept top-1 equality alone | It discards observable full-distribution differences and is weaker than the aligned exact criterion. |
| Patch the GDN fallback or upgrade torch | This changes the audited backend and violates the step scope/A11. |
| Attribute the effect to CUDA `cumsum` | The available measurements do not establish that causal claim. |

## Consequences

- SPEC-01 is accepted as 9/9 under the aligned in-process identity definition.
- Reports must name the exact reference convention and may not claim
  independent-process CUDA bit-exactness.
- Shape-specific warmup and stability proof are part of the versioned config,
  protocol hash, and every decisive artifact.
- SPEC-02 may now build the formal blocking gate and measured cross-path
  tolerances without reopening the settled wrapper-equivalence investigation.

## Evidence

- Experiment: `EXP-0008`.
- `reports/step1_decode_diagnostics.md`.
- `reports/step1_runner_state_diagnostics.md`.
- `reports/step1_formic_hf_divergence_conclusion.md`.
- `scripts/step1_decode_diagnostics.py`.
- `scripts/step1_runner_state_diagnostics.py`.
- Audit: `audits/qwen3_8_27b/06_gated_deltanet_audit.md`.
