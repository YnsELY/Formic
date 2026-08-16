# ADR-0003 — torch 2.4 × transformers 5.8 custom-op annotation shim

- **Status:** ACCEPTED
- **Date:** 2026-08-16
- **Step:** part 1 / step 1
- **Deciders:** step-1 implementation
- **Supersedes / superseded by:** —

## Context

`import transformers` fails on this environment:

```text
ValueError: infer_schema(func): Parameter input has unsupported type
torch.Tensor ... Got func with signature
(input: 'torch.Tensor', weight: 'torch.Tensor', offs: 'torch.Tensor')
```

Transformers 5.8 registers `transformers::grouped_mm_fallback` (a MoE helper) with
postponed — i.e. string — annotations. torch 2.4's `infer_schema` does not resolve
them and rejects the registration at import time.

The checkpoint audit hit the same wall and recorded the same resolution:

> "PyTorch 2.4 ne sait pas inferer une annotation differee d'un custom op MoE de
> Transformers 5.8; les scripts dynamiques resolvent cette annotation au runtime.
> Ce correctif ne change ni le graphe Qwen, ni les poids, ni les calculs."
> — `audits/qwen3_8_27b/README.md`

Every dynamic result in the audit (baseline logits, cache probes, replay
experiments) was produced with this shim active, so Formic must run under the
same condition for its numbers to be comparable.

## Decision

`formic/backbone/torch_compat.py` wraps `torch.library.custom_op` so that a
function's annotations are resolved with `typing.get_type_hints` before schema
inference. It is applied once, on `import formic`, and only when
`torch.__version__` starts with `2.4`.

Its status (`torch_version`, `annotation_shim_needed`, `annotation_shim_applied`)
is included in `environment_report()`, therefore in every run record: no
measurement can be quoted without disclosing that the shim was active.

## Audit constraints engaged

None of A1–A12 are touched. The shim acts on the torch↔transformers registration
interface, not on Qwen modules:

- Qwen3.5 contains **no MoE layer**, so `grouped_mm_fallback` is never called on
  any Formic path; only its *registration* must not raise.
- No Qwen class, function or tensor is modified — the A11 guard test
  (`tests/test_no_cell_reimplementation.py`) still passes, and it deliberately
  targets `Qwen3_5*` / `modeling_qwen3_5` patching, which this is not.

## Alternatives considered

| Option | Why not |
|---|---|
| Upgrade torch | Changes kernels and numerics, and breaks comparability with every audited measurement — the audit's baseline is torch 2.4.1+cu124. |
| Downgrade transformers | The checkpoint declares `transformers_version 5.8.0.dev0`; older versions do not ship the `qwen3_5` implementation. |
| Patch the installed transformers file | Mutates a shared site-packages file: invisible, unversioned, and lost on any reinstall. |
| Catch the error and continue | The failure happens at import time inside transformers' own import graph; there is nothing to catch usefully. |

## Consequences

- Formic runs on exactly the audited stack (torch 2.4.1+cu124 / transformers
  5.8.0 / accelerate 1.14.0 / safetensors 0.8.0), with the same fast-path
  situation (no FLA, no `causal-conv1d` ⇒ exact PyTorch fallbacks, eager
  attention), so audit reference numbers remain directly comparable.
- If the stack is ever upgraded, the shim becomes a no-op automatically
  (`annotation_shim_needed=False`) and the environment report will show it — that
  change must then be treated as a new numerics baseline (plan 2.4).

## Evidence

- Failure and resolution reproduced in step-1 logs (`logs/step1_acceptance.log`).
- `python -m formic.cli env` reports the shim state alongside torch/transformers
  versions and fast-path availability.
