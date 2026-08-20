# SPEC-01 runner call/state diagnostics

Diagnostic only. No runner correction or causal attribution is made here. 
SPEC-01 remains 8/9 and ADR-0004 remains PROPOSED.

## Protocol

- Config hash: `ac4b4adfaa98d5454d57853ddd2d51f419cab56d9aabbafb5505bc9994f44634`
- Prompt-set hash: `995c26d31e99faf8fb0902150ab169c4df2132910053f004e29b3043e469c7d6`
- Text only, batch 1, BF16, six warmups and two measured traces.
- A naked/observed bit-inertness gate ran first on all three paths and all six prompts.
- State hashing then ran as a separate naked/captured gate on runner and explicit HF.
- Boundaries: `prefill` produces legacy logit step 0; `after_forced_N` produces legacy step N+1.
- Protocol identity: `908315a3694632169e9909e7c354d89c65d58d8ae9bf9e0adfb7d7aa5c98e179`.

## Observer gate

Result: **PASS**.

| Prompt | Path | Observed vs naked |
|---|---|---|
| `audit_echo` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `audit_echo` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `audit_echo` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `plain_text` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `plain_text` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `plain_text` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_completion` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_completion` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_completion` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_bugfix` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_bugfix` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_bugfix` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_short` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_short` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_short` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_scope` | `formic_runner` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_scope` | `hf_generate` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_scope` | `hf_explicit` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |

## Call arguments

| Prompt | Runner vs explicit | Generate vs explicit |
|---|---|---|
| `audit_echo` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |
| `plain_text` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |
| `code_completion` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |
| `code_bugfix` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |
| `instruction_short` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |
| `instruction_scope` | 0 differences; first `none` | 213 differences; first `calls[0].additional_arguments.return_dict` |

The runner and explicit HF loop therefore present the same recorded arguments at
all 96 forwards: presence versus absence, effective defaults, tensor shapes,
dtypes, strides, raw-content hashes, cache type and per-layer cache lengths all
match. This is an observed equality of the requested call records, not a causal
claim about later numerical differences.

## Cross-path logits

These comparisons use the final observed traces after the observer gate passed.

| Prompt | Runner / explicit | Runner / `generate()` | `generate()` / explicit |
|---|---|---|---|
| `audit_echo` | 2/16 exact, top-1 4/16, first `after_forced_1`, max delta 14.718750 | 1/16, top-1 3/16, first `prefill`, max delta 13.625000 | 1/16, top-1 5/16, first `prefill`, max delta 14.546875 |
| `plain_text` | 1/16 exact, top-1 4/16, first `after_forced_0`, max delta 12.859375 | 0/16, top-1 2/16, first `prefill`, max delta 13.921875 | 0/16, top-1 2/16, first `prefill`, max delta 14.031250 |
| `code_completion` | 16/16 exact, top-1 16/16 | 0/16, top-1 6/16, first `prefill`, max delta 15.812500 | 0/16, top-1 6/16, first `prefill`, max delta 15.812500 |
| `code_bugfix` | 16/16 exact, top-1 16/16 | 3/16, top-1 5/16, first `prefill`, max delta 12.273438 | 3/16, top-1 5/16, first `prefill`, max delta 12.273438 |
| `instruction_short` | 16/16 exact, top-1 16/16 | 0/16, top-1 4/16, first `prefill`, max delta 13.781250 | 0/16, top-1 4/16, first `prefill`, max delta 13.781250 |
| `instruction_scope` | 16/16 exact, top-1 16/16 | 0/16, top-1 5/16, first `prefill`, max delta 12.718750 | 0/16, top-1 5/16, first `prefill`, max delta 12.718750 |
| **Total** | **67/96 exact, top-1 72/96** | **4/96 exact, top-1 25/96** | **4/96 exact, top-1 27/96** |

The supplied earlier campaign reported runner/explicit at 40/96 exact. This
campaign reports 67/96 under a different invocation schedule: each path has six
naked and six observed warmups, with path order rotated and observer order
alternated. The discrepancy is recorded without attribution. Both naked and
observed final pairs are internally bit-stable.

## State gate

Result: **PASS**.

| Prompt | State-captured vs naked runner | State-captured vs naked explicit |
|---|---|---|
| `audit_echo` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `plain_text` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_completion` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `code_bugfix` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_short` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |
| `instruction_scope` | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 | 16/16 exact, top-1 16/16, first=None, max delta=0.000000e+00 |

## First divergences

| Prompt | First logit | First state | Component | State precedes logit |
|---|---|---|---|---|
| `audit_echo` | `after_forced_1` | `after_forced_1` | `layer 49 / recurrent_states` | False |
| `plain_text` | `after_forced_0` | `after_forced_0` | `layer 49 / recurrent_states` | False |
| `code_completion` | `None` | `none` | `none` | False |
| `code_bugfix` | `None` | `none` | `none` | False |
| `instruction_short` | `None` | `none` | `none` | False |
| `instruction_scope` | `None` | `none` | `none` | False |

No first state divergence precedes the first logit divergence. For
`audit_echo`, states are exact through `after_forced_0`; the first state and
logit differences both occur at `after_forced_1` (legacy logit step 2). For
`plain_text`, both first occur at `after_forced_0` (legacy logit step 1).

At each of those first divergent boundaries, the first differing component in
layer order is `recurrent_states` in GDN layer 49. The differences visible at
that same boundary are:

- GDN layer 49: `recurrent_states`.
- GDN layers 50, 52, 53, 54, 56, 57, 58, 60, 61 and 62: `conv_states` and `recurrent_states`.
- Full-attention layers 51, 55, 59 and 63: K and V.

All components in layers 0-48 are exact at the first divergent boundary. The
complete component-by-component list for every later boundary is retained in
`state_gate.json`. `rope_deltas` is absent from this text-only CausalLM object
on every captured boundary in both paths.

## Generate convention

The runtime calls distinguish `generate()` from the explicit loop as follows:

| Field | Explicit loop | `generate()` |
|---|---|---|
| Prefill cache | Explicit `None`; stock model creates it | Pre-created `DynamicCache`, 64 layers, sequence length 0 |
| `attention_mask` | Absent; effective default `None` | Present; all-ones `[1, prompt_length]`, then grows by one per token |
| `position_ids` | Absent; effective default `None` | Present `[1, prompt_length]` at prefill, then `[1, 1]` |
| `logits_to_keep` | Absent; effective default `0` | Present, value `1` |
| `cache_position` | Unsupported/absent | Unsupported/absent |
| `num_logits_to_keep` | Unsupported/absent | Unsupported/absent |
| `return_dict` | Absent | Present, `True` |
| `use_cache` | Present, `True` | Present, `True` |

`input_ids` contents, dtypes and shapes match at corresponding calls. The
`logits_to_keep=1` call computes the LM head only for the final hidden position;
the explicit default `0` computes it for all positions and slices the final
position afterward. In this measured campaign, `generate()`/explicit diverges
at `prefill` on all six prompts, with 4/96 exact logits overall. This documents
the convention difference and does not assign causality to any one field.

The exact runtime records and all 213 per-prompt field differences are in
`artifacts/step1/runner_state_diagnostics/observer_gate.json`.

## Environment

- Torch: `2.4.1+cu124`
- CUDA: `12.4`
- GPUs: `[{"name": "NVIDIA A40", "total_memory": 47697690624, "capability": [8, 6]}]`
- `pip freeze` SHA-256: `630242aa0441d48bf07e4593cc71847b29f955e4deb831fb2bf9037dba9f63ab`
- Full `pip freeze` output: `observer_gate.json` and `state_gate.json`, field `metadata.pip_freeze.stdout`.
