# ADR-0005 — Formal identity protocol and measured tolerances

- **Status:** PROPOSED
- **Date:** 2026-08-21
- **Step:** part 1 / step 2
- **Deciders:** Yanis validated the horizon-8, corpus-freeze, and 64-token-probe subdecisions; final measured tolerances remain pending
- **Supersedes / superseded by:** —

## Context

ADR-0004 accepts the aligned in-process explicit CausalLM loop as the identity
reference and classifies independent-process CUDA bit-exactness as an
out-of-criterion backend limitation. SPEC-02 must turn that conclusion into a
blocking gate while measuring the separate numerical effects of segmentation,
cached versus recomputed decoding, and snapshot/restore.

The A40 diagnostics performed before calibration refined the meaning of
"aligned". The complete balanced crossover
`balanced-crossover-2026-08-24-r2` measured 1,536/1,536 exact endpoint
comparisons at the same round-relative crossover slot, with zero delta and
zero KL. At the same time, 336/384 raw ordinal groups changed and the raw
RR/NN controls were not uniformly stable. These are measurements only; no
root cause is assigned. They show that raw fingerprints from distinct
process-lifetime ordinals cannot be made a blocking substitute for the
matched endpoint contrast already accepted by ADR-0004.

The earlier schedule matrix r6 also established that the alternating calendar
completed without OOM, returned to its post-load allocation after warmup, and
made all three RR/NN/RN controls stable over their last two repetitions. This
calendar is therefore retained for the economical reference floor. The
official wrapper gate uses the stronger four-round Latin ABBA crossover: every
RR/NN/RN/NR treatment occupies each of the four configuration ordinals before
a contrast is admitted.

The recurrent GDN state is persisted in BF16 and is updated in place. A
segmentation boundary can therefore change rounding even when the mathematical
token sequence is unchanged. Any bounded threshold must be derived from
repeated measurement, never selected from intuition or a single run.

## Proposed decision

The reference is the explicit text-only `Qwen3_5ForCausalLM` loop. Wrapper
identity is decided from matched round-relative endpoint contrasts in one
process. Actual process-lifetime ordinals remain recorded and are explicitly
distinct; the report must never describe them as equal. Raw cross-ordinal
fingerprints remain non-blocking diagnostics. `generate()` is not an identity
oracle.

The last-two matched signature contains only endpoint-difference invariants:
exactness, delta, KL, top-1 agreement and first differing coordinate/value. It
does not contain the absolute token ID or an endpoint's raw fingerprint; those
would reintroduce the cross-ordinal condition that the crossover was designed
to control.

Numerical calibration is a separate question from wrapper identity. Segmented
prefill is compared against stock full-prefix recomputation at every segment
boundary. Cached decode is compared against stock full recomputation for short
and medium prompts. Long cached decode retains a stock cached reference because
long full recomputation is outside the validated plan. This separation ensures
that the tolerance campaign actually measures segmentation and BF16 recurrent
state effects instead of comparing two copies of the same path.

Four execution paths are covered: monolithic prefill, segmented prefill,
cached decode, and full-recomputation decode. Greedy and fixed-seed sampling
are crossed where applicable. Sampling continuations are produced once by the
reference and then forced identically on every compared path.

The calibration horizon and the blocking CI/GPU-gate horizon are both exactly
eight decode frames. This equality is a schema-validated invariant, not a CLI
default that a run may override. The measured SPEC-01 effects appear in the
first decode steps, so eight frames retain the observed onset while reducing
the required 27B forwards. A separate logits-only accumulation probe runs for
64 frames on the pinned short and medium prompts. It is a long-range diagnostic:
any growth is reported numerically without a causal attribution or an invented
threshold.

For each exact input length, six traces warm the path with no state capture.
Six warmups are shared once per exact input shape and process. Three traces are
then measured per tolerance configuration. The exact legacy continuity gate
uses the two repetitions required to assert last-two stability, outside
calibration. The last two measured traces or matched contrast signatures must
be bit-exact or the case is invalid.

Inference measurements do not vary the RNG seed: every compared forward uses
a forced continuation, stock inference has no active dropout, and token
selection is outside the measured forward. A weight-free stock-toy control ran
the same forced cached continuation after seeds 101 and 909 and obtained
bit-exact logits at every step. Seeds 0, 1, and 2 remain only for generating
the candidate sampled continuations from the reference; once generated, those
token IDs are committed and forced through every measured path. If the real
checkpoint contradicts the toy control, the campaign records the values and
stops without assigning a cause.

Short and medium prompts capture residual hidden states at all 17 group
boundaries plus the state of the group completed at its natural exit boundary.
Long prompts capture logits and final state only. Output logits are measured at
the model output only. GDN/KV are `not_applicable` for full recomputation with
no cache.

Long prompts omit full-recomputation decode and retain only median and regular
quarter segmentations. Early and late single cuts, and cached-versus-recompute
measurements, remain covered by short and medium prompts.

Both prompts per class retain full and segmented prefill. Decode calibration is
limited to one pinned prompt per class: `short_error_assertion`,
`medium_cache_regression`, and `long_resume_incidents`. The accumulation probe
uses the first two of these. Snapshot/restore continuity uses `audit_echo`; its
metrics are captured before the calibration classes, then adjudicated against
the newly materialised tolerances without rerunning the model.

The default criterion is exact. Where repeated measurement proves a bounded
criterion necessary, the threshold is twice the maximum observed delta across
the three repetitions, and never below the reference/reference floor. The
floor is measured logits-only on `audit_echo`, the pinned short prompt and the
pinned medium prompt under the alternating RR/NN/RN schedule. The maximum RR
delta is propagated into every logits candidate row; it is never hard-coded to
zero during promotion.
Blocking metrics are 100% top-1 agreement at aligned protocol and maximum
absolute delta. KL is recorded but non-blocking. Tolerances are keyed by length
class while evidence retains every exact input length.

The gate records the first divergent decode step, the first
boundary/layer/component, and the first tensor coordinate. Interruption-safe
artifacts are written atomically per prompt and may be resumed only when all
protocol hashes still match.

The twelve-prompt corpus is frozen as schema v2 at corpus SHA-256
`482e63d88a53d2850fe87db648f7d6fe2414ca5ee64b1a307de7cb3501c1f3c0`.
It contains exact rendered text, exact token IDs and separate hashes for both.
Any future change requires an ADR and invalidates prior verdicts.

The A40 preflight writes measured path timings, the single model-load duration,
and a per-phase total-duration estimate. The report is informational: it takes
no budget argument, returns success, and the session continues automatically.
The planned session is one process and one complete model load. The official
launcher pins the A40 placement cap to 35 GiB, the value under which the r6 and
r2 diagnostics completed without OOM. It verifies the loaded 851-tensor
backbone against committed hash
`74e1813c29b065406f4b772ed7c9059b8455428bff9aa6e572645cf09743c662`
before the first identity measurement.

The revised plan contains 8,549 model forwards: 333 preflight, 120 trace
inertness, 3,552 legacy crossover, 624 noise floor, 48 snapshot/restore, 96
reference continuation generation, 1,728 main calibration and 2,048 for the
64-frame cached-versus-recompute probe. The historical short-shape sensitivity
projection is 7.31 hours; the real post-preflight estimate supersedes it and
remains informational.

No numerical tolerance is proposed in this ADR. `tolerances.json` will be
created only by the final A40 calibration campaign.

## Audit constraints engaged

- **A1** — recomputation supplies no cache; no `use_cache=False` call is used
  to protect a cache from mutation.
- **A2** — any explicit hybrid cache is constructed with `model.config`.
- **A3** — snapshot/restore never calls `crop()`.
- **A4** — capture and every restoration deep-clone recurrent state; fork
  isolation is checked by storage identity and mutation probes.
- **A5** — no norm or cell equation is copied or changed.
- **A6** — model-attached state is captured separately from the cache; absence
  of `rope_deltas` in text-only mode is explicit.
- **A8** — batch 1, no padding.
- **A9** — attention K/V are copied exactly as stored; no reinterpretation.
- **A10** — MTP remains inactive and excluded.
- **A11** — only stock modules plus read-only boundary observers are used.
- **A12** — real loads use actual shard headers and strict post-load matching;
  local CI uses the committed audited header manifest only for weight-free
  structural guards.

## Alternatives considered

| Option | Why not |
|---|---|
| A priori epsilon | Violates the measured-tolerance requirement. |
| Average across seeds/repetitions | A blocking identity gate must cover the measured worst case. |
| KL as a blocking metric | KL remains diagnostic; top-1 plus delta gives the accepted gate semantics. |
| Fixed-order RR/NN/RN/NR quartet | It falsely conflates four configuration ordinals; the measured ordinal effect requires Latin rotation before matching. |
| Raw RR/NN fingerprint stability as wrapper verdict | The r2 evidence compares distinct process-lifetime ordinals; matched endpoint contrasts are the interpretable identity evidence. |
| Full cache copied at all 17 boundaries | Redundant and too costly; each group state is captured once at its natural boundary. |
| Full boundary capture for 2k–4k prompts | Excessive memory and transfer cost; long prompts retain logits plus final state. |
| Full-recomputation decode for 2k–4k prompts | Repeats a complete long prefill at every token; short and medium retain this comparison. |
| Early/late segmentation for 2k–4k prompts | The two cheaper long segmentations retain boundary-count information; all four remain on short/medium. |
| 3 seeds × 3 measured repetitions | The measured forward consumes fixed IDs and has no RNG use; seed variation is isolated to reference continuation generation. |
| Sixteen-frame calibration/gate horizon | It did not fit the forward budget; observed effects already occur in early steps and the separate 64-frame probe retains long-range visibility. |
| Different CI and calibration horizons | A gate cannot apply thresholds outside the horizon that justified them. Both are pinned to eight. |
| Inspect ephemeral no-cache state inside cells | Requires cell instrumentation and violates A11. |

## Consequences

- CI can distinguish exact wrapper identity from measured cross-path numerical
  equivalence.
- A tolerance change must cite its raw calibration evidence, ADR, and report.
- A changed config or 851-tensor backbone hash invalidates the last PASS
  verdict until the manual A40 gate is rerun.
- The 64-frame probe is reported separately and cannot silently widen an
  eight-frame tolerance.
- The final threshold table remains pending measurements; the preflight timing
  report is written before any identity measurement begins.
- A completed diagnostic can be reassessed without mutation by
  `scripts/step2_reassess_crossover.py`; the source artefact and its hashes stay
  immutable.

## Evidence

- Existing baseline: `EXP-0008`, ADR-0004, and step-1 diagnostic reports.
- A40 schedule matrix r6: six complete configurations, no OOM; alternating
  RR/NN/RN controls stable over the recorded repetitions.
- A40 balanced crossover r2: 1,536/1,536 same-slot endpoint comparisons exact;
  336/384 raw ordinal groups changed; diagnostic status `COMPLETE`.
- Weight-free seed control: `tests/test_forced_continuation.py`, seeds 101/909,
  same forced continuation, exact logits at every step.
- Planned calibration experiment: to be allocated before the A40 session.
- No tolerance-calibration measurement or official SPEC-02 verdict exists yet.
