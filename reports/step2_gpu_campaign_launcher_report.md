# SPEC-02 — GPU campaign launcher (local implementation report)

**Status:** implementation complete; no A40 measurement has been run by this
change.

## Delivered

- `scripts/step2_a40_campaign.py` launches a single-process, single-load A40
  calibration run.
- `formic.science.identity.campaign_plan` pins the validated Option-B plan:
  horizon 8, long-prompt reductions, 64-frame probe and 4,139 planned forwards.
- The preflight performs the strict load/inventory path, hashes the 851-tensor
  backbone, times the 18 approved paths (one dry plus two timed traces),
  measures representative GPU-to-CPU transfer bandwidth, writes
  `preflight/estimate.json`, displays the informational estimate, then
  continues without a budget cut-off.
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

## Local verification

- Full weight-free test suite: **312 passed**.
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
   the single sampled decode variant in the fixed 4,139-forward plan.
2. Launch the campaign on the A40, inspect its terminal artefact, and stop the
   pod before analysis.
3. Review candidate bounded rows, provide their physical justifications,
   promote and version the tolerance table, report, governance record and
   official GPU verdict. ADR-0005 remains **PROPOSED** until explicit human
   acceptance.
