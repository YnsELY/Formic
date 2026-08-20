# SPEC-01 — Formic foundation and Qwen3.8-27B backbone integration

**Branch:** `step-1-backbone-integration`

**Status:** preliminary verification **FAILED (8/9)**; human validation pending

**ADR-0002:** PROPOSED

**Formal identity gate:** deferred to SPEC-02

## Delivered

| Area | Evidence |
|---|---|
| Strict text-only BF16 loader | `formic/backbone/loader.py`, `configs/default.yaml` |
| Strict inventory and MTP/vision exclusions | `formic/backbone/inventory.py`, `tests/test_inventory.py` |
| Bijective key rename | `text_only_name_mapping()`, mapping report, 851-record test |
| 16-group view over stock HF cells | `formic/backbone/groups.py`, `tests/test_groups.py` |
| 17 config-driven boundaries | `formic/backbone/boundaries.py`, `configs/step1_noop_hooks.yaml` |
| Batch-1/no-padding boundary | `formic/backbone/runner.py`, `tests/test_runner.py` |
| Reproducibility policy | `formic/science/determinism.py`, `tests/test_determinism.py` |
| Preliminary acceptance | `scripts/step1_acceptance.py`, `artifacts/step1/` |
| Governance | `STATUS.md`, `experiments/`, `docs/adr/`, `docs/conventions.md` |

No stop/resume execution, partial-depth path, sidecar, adapter, MTP runtime,
training, new token, transaction runtime, or active multimodal path was added.

## Measurements

The last clean preliminary run uses config hash
`19455e2b0c639dd9c9de967a4566f743d59dc4a241944053511b63c2a97a8ef2`
and frozen prompt-set hash
`995c26d31e99faf8fb0902150ab169c4df2132910053f004e29b3043e469c7d6`.
It was executed from clean implementation commit
`4e99e9a92f2c0adb95bee1afb4d16150f97d6dc3` and registered as `EXP-0007`
(artifact-set SHA-256 `e4003397963a531d8318f92a7ebd41eca64d46680c433df303602ab7e1476448`).

### Strict load and memory

```text
class                    Qwen3_5ForCausalLM
parameters               26,895,998,464 (50.10 GiB BF16)
matched tensors          851 / 851
missing / unexpected     0 / 0
shape / dtype mismatch   0 / 0
HF missing / unexpected  0 / 0
HF mismatch / errors     0 / 0
declared exclusions      MTP: 15 tensors; vision: 333 tensors
vision module present    false
MTP module present       false
CUDA parameter bytes     39,839,137,984
CPU-offloaded bytes      13,952,858,944
```

The 460,730,096 vision parameters are excluded before construction. The loaded
parameter delta from the audited multimodal tree is exactly the audited vision
parameter count.

### Structural key rename

```text
source tensors       851
target tensors       851
renamed              850 (model.language_model.* -> model.*)
unchanged              1 (lm_head.weight)
injective            true
onto expected set    true
inverse roundtrip    true
metadata preserved   true
```

The test applies the actual regex to every source name, inverts all target names,
and preserves each record's shape, BF16 dtype, and parameter count. This proof is
independent of the Formic-vs-HF logit comparison.

### Formic vs direct Hugging Face CausalLM

At each prompt's evaluated next-token position:

| Metric | Result |
|---|---:|
| FP32 logit SHA-256 equality | 6/6 |
| Maximum absolute logit divergence | 0.000000e+00 |
| Mean KL(ref || Formic) | 0.000000e+00 nats/token |
| Maximum KL(ref || Formic) | 0.000000e+00 nats/token |
| Top-1 agreement | 6/6 |

Cached decode is not identical:

| Path | Exact runs |
|---|---:|
| Explicit greedy cache loop | 0/4 |
| Native greedy `generate()` | 0/6 |
| Native sampled `generate()` | 0/3 |

The divergence occurs after an identical first-token prediction. Strict PyTorch
determinism fails at the stock GDN fallback's prefill cumsum:

```text
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
```

This kernel is a confirmed nondeterministic operation on the recurrent-cache
construction path; the probe alone does not prove it is the sole cause of every
token divergence. Replacing or patching it would modify the stock cell/backend
and is forbidden by A11 and SPEC-01. The generation checklist item remains
failed rather than being weakened silently.

### Candidate warmup-protocol rerun

An exploratory rerun on 2026-08-18 used candidate config hash
`ac4b4adfaa98d5454d57853ddd2d51f419cab56d9aabbafb5505bc9994f44634` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` set before each process imports Torch,
TF32 disabled, Flash and memory-efficient SDPA disabled, and math SDPA enabled.
For every one of the six prompt/cache shapes, each implementation performed six
unmeasured fresh-cache traces followed by two measured traces.

| Check | Result |
|---|---:|
| Formic last-two measured traces exact | 6/6 prompts |
| Direct HF last-two measured traces exact | 6/6 prompts |
| Formic/HF prefill logit SHA equality | 6/6 prompts |
| Explicit greedy cache loop | 0/4 |
| Native greedy `generate()` | 0/6 |
| Native sampled `generate()` | 0/3 |

The protocol establishes within-process stability, but not equality between the
two independently initialized CUDA processes. The run was intentionally not a
formal acceptance: its implementation worktree was dirty, so `--stage compare`
correctly refused to emit a verdict. It reinforces the existing 8/9 failure;
it does not create an `EXP` registry result or start SPEC-02.

### Real-checkpoint boundary inertness

The hook stage loads the real checkpoint once, computes six baseline forwards,
registers all 17 no-op insertion hooks, and recomputes the same forwards in the
same process.

| Check | Result |
|---|---:|
| Registered hooks during proof | 17 |
| Logit SHA-256 equality | 6/6 |
| `torch.equal` on logits | 6/6 |
| Maximum absolute delta | 0.000000e+00 |
| Hooks after detach | 0 |

## Audit Constraints

| Code | Treatment in SPEC-01 |
|---|---|
| A1 | `use_cache=False` is used only without a supplied cache. Cached generation uses a fresh model-created cache and never treats the flag as read-only protection. |
| A2 | Formic never constructs `DynamicCache`; the stock model creates it with its own config. |
| A3 | No `Cache.crop()` call or rollback mechanism exists. |
| A4 | No restore/fork exists and no recurrent buffer is shared between consumers. |
| A5 | No normalization cell or formula is reimplemented. |
| A6 | Formic does not read, write, or classify `rope_deltas` as cache state. |
| A7 | Only `Qwen3_5ForCausalLM` is loadable; the vision tower is absent from the module tree. |
| A8 | Runner guards enforce batch 1 and reject padded masks. |
| A9 | Formic does not inspect or transform K/V cache contents. |
| A10 | All 15 MTP tensors are named and counted as strict exclusions. |
| A11 | Groups are a view, hooks touch only the residual stream, and AST/source guards reject cell copies, subclasses, or monkeypatches. |
| A12 | Header inventory precedes loading; Transformers `loading_info` is fatal outside declared exclusions; post-load names, shapes, dtypes, and counts are compared both ways. |

## Exit Checklist

| Item | Result | Evidence |
|---|---|---|
| Native greedy and sampled generation identical to direct HF (preliminary) | **FAIL** | Prefill 6/6 exact; cached generation 0/13 exact; nondeterministic stock CUDA cumsum diagnosed |
| Group/layer mapping conforms | PASS | 16 groups, 64 layers, attention at 3, 7, ..., 63 |
| No cell code reimplemented or modified | PASS | A11 guard suite |
| Strict inventory; permissive loading impossible; MTP explicit | PASS | 851/851, bijection, declared exclusions |
| Vision tower not constructed | PASS | module absent; 50.10 GiB loaded; audited 460,730,096-param delta |
| 17 registered boundaries are bit-inert on real logits | PASS | SHA and `torch.equal` 6/6 |
| Execution fully described by YAML | PASS | default and no-op-hook config hashes recorded |
| STATUS, experiment registry, ADR template, conventions | PASS | repository governance files |
| Report covers measurements, A1-A12, and deviations | PASS | this document |

**Preliminary SPEC-01 result: FAILED (8/9).** This is not the SPEC-02 identity
gate. No blocking tolerance or formal identity claim has been introduced.

## Deviations And Open Decision

- Repository modules live under the importable `formic/` package rather than as
  unrelated top-level Python modules; ADR-0001 records this layout.
- The audited torch 2.4 / transformers 5.8 compatibility shim remains required
  for importing transformers; ADR-0003 records it.
- Quantized loading was not implemented, per validated scope; SPEC-01 is BF16.
- Stop/resume at group boundaries was not implemented, per validated scope; only
  the 17 inert boundaries exist.
- ADR-0002 remains PROPOSED pending human validation.
- ADR-0004 is PROPOSED; no statistical criterion has been accepted. Follow-up
  `EXP-0008` establishes exact aligned CUDA Formic/HF decode, exact CPU
  Formic/HF decode, a deterministic first-execution effect, unchanged model
  tensor state, and exact post-warmup repeats; see
  `reports/step1_decode_diagnostics.md`. The candidate `N=6` warmup acceptance
  rerun stabilizes each process but retains 0/13 Formic/HF generation equality
  across processes. This does not retroactively turn the preliminary generation
  checklist green: SPEC-01 remains 8/9, and SPEC-02 must not start while
  SPEC-01 remains unvalidated.
