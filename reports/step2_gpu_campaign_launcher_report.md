# SPEC-02 — GPU campaign launcher (local implementation report)

**Status:** official launcher adapted to the measured A40 scheduling evidence;
the tolerance calibration and official verdict remain incomplete.

## Delivered

- `scripts/step2_a40_campaign.py` launches a single-process, single-load A40
  calibration run.
- `formic.science.identity.campaign_plan` pins the validated Option-B plan:
  horizon 8, long-prompt reductions, 64-frame probe and 8,549 planned forwards.
- The exact legacy gate is a four-round Latin ABBA crossover. Every RR/NN/RN/NR
  treatment occupies each configuration ordinal before endpoint contrasts are
  compared. Raw process-lifetime fingerprints remain diagnostic only.
- The economical noise floor uses the alternating r6 calendar for RR/NN/RN and
  propagates the measured RR maximum into every logits tolerance candidate.
- Calibration compares segmented prefill with full-prefix stock recomputation,
  and cached decode with full stock recomputation for short/medium prompts.
- The preflight performs the strict load/inventory path, hashes the 851-tensor
  backbone, times the 18 candidate paths and the 12 distinct canonical
  companions (one dry plus two timed traces),
  measures representative GPU-to-CPU transfer bandwidth, writes
  `preflight/estimate.json`, displays the informational estimate, then
  continues without a budget cut-off.
- The launcher invokes `scripts/step2_budget_gate.py` as the informational
  reporter, verifies that its JSON equals the in-process estimate, and then
  continues; the reporter has no `--budget-hours` and never returns a NO-GO.
- Transient CUDA workspaces are collected and inactive allocator blocks are
  released after each complete preflight path and at every phase boundary;
  cleanup is never performed between measured repetitions.
- Phase and case artefacts use atomic replace plus a source-hash manifest;
  `--resume` rejects changed commit/config/corpus/backbone sources.
- The run order is preflight, trace inertness, legacy continuity, noise floor,
  snapshot/restore, sampled-continuation generation, short/medium/long
  calibration, then the 64-frame probe.
- A first campaign writes raw evidence and a review-required tolerance
  candidate. It terminates with `CALIBRATION COMPLETE — PROMOTION REQUIRED`,
  never a fabricated official PASS.
- `scripts/step2_promote_calibration.py` requires explicit human
  justifications for bounded rows before it can materialise a strict
  `tolerances.json`. It does not alter ADR status, governance, verdicts or Git.
- The loaded 851-tensor backbone must match the committed audit hash before the
  first measured gate. The launcher defaults the A40 placement cap to 35 GiB,
  matching the successful r6/r2 diagnostic runs.

## A40 incident follow-up

- Run `a40-2026-08-22-r2` exhausted CUDA memory before its first measured
  comparison. The follow-up records allocator headroom around every preflight
  path and disables autograd in control-only forwards.
- Run `a40-2026-08-22-r3` reached the legacy continuity phase without an OOM,
  then stopped with `InvalidMeasurement` because the final two
  `legacy__audit_echo` traces were not stable. This is a measured observation,
  not a root-cause attribution.
- Each measured repetition now atomically updates a non-authoritative
  diagnostic artefact. If stability or an exact gate fails, the terminal error,
  per-repetition reference/runner fingerprints, metrics, and failure memory
  observation remain available even though the case is not marked complete.
- `scripts/step2_legacy_stability_probe.py` is a deliberately narrow A40
  diagnostic control for that one pinned legacy case. It is not a calibration
  or an identity verdict.
- Schedule matrix r6 completed all six configurations without OOM. The
  alternating RR/NN/RN paths were stable over the recorded repetitions;
  sequential RR retained a raw ordinal change.
- Balanced crossover r2 completed 3,072 measured forwards without OOM. It
  observed 1,536/1,536 exact same-slot reference/runner comparisons while
  336/384 raw ordinal groups changed. No cause was assigned.
- The previous readiness rule incorrectly required raw outputs from distinct
  process-lifetime ordinals to match. The launcher now blocks on exact matched
  endpoint contrasts and their last-two signatures, while preserving all raw
  controls in the artefact as non-blocking diagnostics.

## Local verification

- Full weight-free test suite: **358 passed**.
- A11/A12/inert-boundary guards: **182 passed**.
- `formic verify`: PASS; `identity-check --toy`: PASS.
- The campaign-plan, preflight-estimator, candidate/promotion, artefact-resume
  and greedy forced-continuation paths are covered by dedicated tests.
- No checkpoint was loaded, no GPU forward was run and no threshold was
  selected in this implementation session.

## Audit constraints A1–A12

| Constraint | Treatment in the launcher |
|---|---|
| A1 | Cacheless calls are never used as cache protection; each measured cache path owns a fresh cache. |
| A2 | Every explicit cache uses `DynamicCache(config=model.config)`. |
| A3 | The launcher does not call `crop()`; snapshot restoration uses the existing audited primitive. |
| A4 | The real-checkpoint snapshot phase checks storage identity between snapshot, branch A and branch B, then mutation isolation. |
| A5 | No cell or norm equation is copied; all forwards invoke stock Hugging Face modules. |
| A6 | Snapshot/restore continues to capture model-attached state separately; text-only absence is explicit in the primitive. |
| A8 | Inputs are constructed strictly as batch 1 without padding. |
| A9 | Snapshot and trace layers retain K/V as supplied by the audited cache. |
| A10 | The loader remains text-only and excludes the inactive MTP head. |
| A11 | The runner uses the existing group view and read-only `IdentityTraceCollector`; no Qwen cell is replaced. |
| A12 | The strict inventory executes before loading; the streaming 851-tensor hash is generated from that validated inventory. |

## Remaining human actions before / after the pod

1. Select which generated sampled continuation seed (`0`, `1` or `2`) will be
   the single sampled decode variant in the fixed 8,549-forward plan.
2. Launch the campaign on the A40 and inspect its incremental and terminal
   artefacts. The launcher neither stops nor asks to stop the pod.
3. Review candidate bounded rows, provide their physical justifications,
   promote and version the tolerance table, report, governance record and
   official GPU verdict. ADR-0005 remains **PROPOSED** until explicit human
   acceptance.
