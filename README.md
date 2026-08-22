# Formic

**Formic** is an execution model specialized for software engineering. It is the
neural execution component of **Uly Code**, an agentic system designed to carry
out software-engineering tasks reliably, controllably, and with verifiable
results.

Formic is not initially intended to be a general-purpose autonomous
conversation assistant. It is designed to receive a task that has already been
understood, diagnosed, and scoped by Uly Code, then turn that task into a series
of correct, bounded, and verified software actions.

## Uly Code and Formic

Uly Code is the overall agentic system. It separates two responsibilities that
are often combined in current coding agents:

```text
Frontier orchestration model
    understand the problem
    investigate the repository
    diagnose the cause
    define the global strategy
    decompose the work
                |
                v
Precisely scoped, constrained, verifiable task
                |
                v
Formic
    understand the scoped task
    read the relevant evidence
    make local decisions
    produce a typed action
    verify and repair
    finish with evidence
```

The orchestration model is responsible for open-ended investigation, difficult
reasoning, discovery, and global strategy. **Formic is the execution model**:
it receives a defined objective, the applicable constraints, the necessary
context, and the success criteria, then executes with discipline.

This separation is the central principle of the project:

- the orchestration model finds and understands the problem;
- Formic executes the task faithfully;
- Uly Code stores state, applies rules, and verifies results;
- no model statement alone is sufficient to mark a task as complete.

Formic is therefore designed, initially and specifically, for the **Uly Code
agentic system**. Its interfaces, constraints, and architecture are defined
around this executor role.

## Training Objective

The purpose of training Formic is to create a model that can perform code
generation and long, complex software-engineering tasks that are **fully and
precisely scoped** at the same practical level as very large frontier models.

The training target is practical parity with frontier models for execution:
Formic should deliver comparable reliability and comparable code and output
quality on a perfectly framed task, even though it is much smaller and much less
expensive to run.

For those well-defined tasks, Formic should approach frontier-model quality in:

- execution reliability;
- code and output quality;
- instruction fidelity;
- strict scope adherence;
- understanding of the provided repository context;
- verification and local error correction;
- long-horizon task completion;
- completion awareness;
- consistency and controllability.

The target is not to build a smaller general-purpose problem solver. Formic is
not intended to discover the solution to an open-ended problem, investigate an
unknown repository from scratch, choose a global strategy, or replace a
frontier-scale reasoning model. Those responsibilities belong to the Uly Code
orchestrator.

The objective is narrower and deliberate: **perform a perfectly framed task as
well as a much larger frontier model, while using substantially less inference
compute and therefore costing much less to run.**

Uly Code combines both capabilities:

```text
Large frontier model
    discovery, investigation, deep reasoning, global planning
                +
Formic
    reliable, high-quality, low-cost execution
                =
Frontier-level software-engineering outcomes
at a substantially lower execution cost
```

The distinction between an orchestrator and an executor is essential. Formic is
not meant to reason less in the sense of being incapable of reasoning. It must
still understand a task, analyze the consequences of a change, reason about
constraints, choose an action, verify its work, and correct errors. Its
reasoning must serve execution rather than open-ended discovery.

## The Target Execution Model

The project is not simply adding a specialized prompt to an existing large
model. The target architecture, called **CAPE-R** (*Contract-Aware Progressive
Executor, Revised*), changes the fundamental unit of work:

```text
Trusted instruction contract
        +
Versioned, typed state
        +
Selected repository evidence
        |
        v
Bounded neural decision
        |
        v
One typed action or state proposal
        |
        v
Deterministic validation and external evidence
        |
        +--> atomic commit
        |
        +--> rejection without side effects
```

A long task becomes a sequence of execution transactions rather than one
conversation that accumulates the entire task history in a single decode.

Each transaction must:

1. read an immutable, versioned instruction contract;
2. read external, typed state;
3. receive only the evidence that it needs;
4. produce one bounded primary action;
5. have that action checked against the system rules;
6. apply it in a controlled environment;
7. verify the result;
8. commit a new state or reject the action without side effects.

Completion is therefore not only a sentence produced by the model. It is a
prediction that must be confirmed by deterministic evidence: tests, parsing,
scope checks, repository state, and any other applicable validation.

## The Four Architectural Planes

The CAPE-R target architecture is organized into four complementary planes.

### Control Plane

The control plane contains the instruction contract (`ContractIR`): objectives,
constraints, authority, scope, success criteria, and applicable rules. A
non-neural reference monitor checks whether proposed actions comply with this
contract. It may reject an action, but it does not invent one.

### State Plane

The state plane, called the **State Fabric**, stores durable information:

- repository snapshots;
- important files, symbols, and relationships;
- the task obligation graph;
- collected evidence;
- previous failures and attempts;
- state transitions;
- an append-only record of decisions and validations.

Durable state does not depend on fragile neural-cache contents or an ever-growing
conversation.

### Execution Plane

The execution plane reuses the Qwen trunk preserved by Formic. It receives the
contract, state, and relevant evidence, then produces a structured decision
through typed outputs and a constrained grammar rather than only free-form text.

In the longer-term architecture, the system may control the amount of neural
computation at natural model boundaries. This capability is deliberately
deferred until the full-depth foundation has been measured and validated.

### Commit Plane

The commit plane checks shape, scope, hashes, tests, parsers, types, and all
other applicable criteria. It applies a modification in a controlled environment
and then atomically commits or rejects the action.

## Why Start from Qwen3.8-27B?

Formic is not pretrained from scratch. The project uses the Qwen3.8-27B
checkpoint as its neural substrate because it provides a strong foundation for
code, reasoning, and agentic workloads.

The checkpoint includes, among other properties:

- approximately 27 billion parameters;
- 64 decoder layers;
- 16 hybrid groups;
- a verified pattern of three Gated DeltaNet layers followed by one full-attention
  layer in each group;
- a very large native context;
- an architecture suitable for studying adaptive computation and sequential
  state.

These properties are not treated as assumptions. The checkpoint was audited
before implementation, and architectural decisions must respect what the audit
actually established.

## Weight-Reuse Discipline

Formic follows a strict invariant:

```text
all Formic mechanisms disabled == original Qwen behavior
```

This means that:

- Qwen cells are not reimplemented;
- Qwen cells are not copied and modified;
- equations, weights, normalization, and residual ordering are preserved;
- tensor loading is strict and checked in both directions;
- the vision tower is not constructed in the text-only path;
- hooks and boundaries are disabled by default;
- every new capability is added around the trunk and placed behind a
  configuration flag disabled by default;
- every quoted measurement retains its environment, configuration, seed, commit,
  and experiment identifier.

This discipline makes it possible to distinguish between:

- the behavior of the base model;
- the capabilities added by the Formic architecture.

## How the Project Will Be Built

Development follows eight strictly ordered steps. A step is not considered
validated without tests, a report, documented status, and human validation where
required.

### Part 1: Foundations

1. **Formic foundation and backbone integration**: repository, configuration,
   scientific registry, strict loading, and a testable view of the Qwen groups.
2. **Identity and numerical tolerances**: blocking baseline, reproducibility
   measurements, and snapshot/restore primitives.
3. **Evaluation and baselines**: frozen suites, metrics, seeds, and reproducible
   baselines.
4. **Full-depth transaction engine**: `ContractIR`, State Fabric, transaction
   engine, and reference monitor.
5. **Typed software actions**: edits, tool calls, state updates, questions,
   abstention, and completion, with validation and sandboxing.
6. **First end-to-end loop and FORMIC-M0**: run small, complete software tasks
   and measure the benefit of the runtime alone.
7. **Episode production**: industrialize labeled execution episodes for training.
8. **First neural sidecars and FORMIC-M1**: add small specialized components,
   keep the trunk frozen, and verify every gain through ablation.

### Part 2: Advanced Capabilities, Deliberately Deferred

Part 2 has not started. It will begin only after a measured and validated
foundation. It may include:

- progressive-depth exits;
- learned routing;
- thinking and scratch budgets;
- a learned "continue reasoning or act" decision;
- speculative decoding;
- controlled GDN state rollback;
- MTP integration;
- long-horizon tasks and outcome/preference training;
- reinforcement learning;
- multi-GPU serving and the vision path.

These capabilities have not been removed from the project. They are sequenced
after the foundations so that advanced mechanisms are not built on an unmeasured
baseline.

## Current Status

SPEC-01 (the first step of Part 1) is complete at 9/9.
SPEC-02, which turns that preliminary evidence into a blocking identity gate
and adds measured tolerances plus snapshot/restore, is in progress. The local
implementation and the one-process A40 campaign launcher are complete; the
final A40 calibration has not yet run.

SPEC-02 is currently pinned to the following protocol:

- the twelve-prompt corpus is frozen in
  [`configs/reference_prompts.yaml`](configs/reference_prompts.yaml), including
  exact rendered text, token IDs, per-prompt hashes, and corpus hash
  `482e63d88a53d2850fe87db648f7d6fe2414ca5ee64b1a307de7cb3501c1f3c0`;
- calibration and CI use an eight-frame decode horizon;
- the long class has no full-recompute decode and uses only median and quarter
  segmentation;
- the 64-frame logits-only accumulation probe remains mandatory for one short
  and one medium prompt;
- the A40 session is one model process and one full checkpoint load;
- the preflight reports load time, per-post duration estimates, and total
  duration, then continues automatically without a budget gate.

### Available Today

- repository structure and scientific conventions;
- strict run-configuration schema;
- `EXP-...` experiment registry;
- text-only BF16 checkpoint loading;
- strict tensor inventory;
- bijective weight mapping;
- the 16-group hybrid view;
- 17 inert boundaries;
- a native generation runner;
- determinism controls and experiment reports;
- in-memory snapshot/restore with synthetic hybrid-cache tests;
- the frozen SPEC-02 prompt corpus and strict identity protocol types;
- the weight-free `identity-check --toy` gate;
- the post-preflight duration reporter in `scripts/step2_budget_gate.py`;
- the resumable, single-load A40 calibration launcher in
  `scripts/step2_a40_campaign.py`;
- weight-free tests and A11/A12 safeguards.

### Backbone Validation Status

The accepted SPEC-01 verification is **9/9**:

- Formic/HF prefill is bit-identical on six prompts;
- strict loading and model structure are validated;
- no-op hooks are bit-inert;
- Formic and HF are exact in several aligned controls;
- Formic and the explicit Hugging Face reference loop are exact at aligned
  in-process execution ordinals;
- runner/reference arguments are identical over 96 forwards and no state
  divergence precedes the logits.

The diagnostics establish a deterministic first-execution/ordinal effect that
is also observable with stock Hugging Face. They do not establish a root cause.
Independent-process CUDA bit-exactness is a documented backend limitation and
is outside the accepted identity criterion. Formic does not claim it.

`generate()` is not used as the reference oracle because it follows a different
call convention, notably `logits_to_keep=1`. The pinned reference is the
explicit `Qwen3_5ForCausalLM` loop. SPEC-02 measures cross-path tolerances under
that aligned in-process convention and makes the identity gate automatic and
blocking. Its duration reporter is informational only and never refuses to
continue on budget grounds.

See [`STATUS.md`](STATUS.md) and the [divergence analysis closing
report](reports/step1_formic_hf_divergence_conclusion.md) for the detailed
measurements, limitations, and recommended decision.

## Quick Start

The following commands verify the project without loading the model weights and
complete in seconds:

```bash
export PYTHONPATH=$PWD

python -m formic.cli verify
python -m formic.cli structure
python -m formic.cli inventory
python -m formic.cli config
python -m formic.cli env
python -m formic.cli identity-check --toy
python -m pytest tests/ -q
```

Commands that load the checkpoint require the local environment and checkpoint:

```bash
python -m formic.cli load
python -m formic.cli generate --prompt "..." --chat
python scripts/step1_acceptance.py --stage all
```

The final A40 calibration is launched only from a clean pod checkout with the
checkpoint mounted at `/workspace/Qwen3.8-27B` and one visible NVIDIA A40. It
loads the model exactly once, runs the preflight first, displays the measured
estimate, and then continues automatically:

```bash
python scripts/step2_a40_campaign.py \
  --run-id a40-YYYY-MM-DD \
  --sampled-continuation-seed <0|1|2>
```

`--resume` reuses atomically completed cases only when the commit, config,
corpus and backbone hashes still match. The first run writes raw measurements,
`tolerances.candidate.json` and `verdict.candidate.json`, then reports
`CALIBRATION COMPLETE — PROMOTION REQUIRED`; it cannot claim an official PASS
before human review and promotion of the measured tolerances. The command stops
on the first hard gate failure. It prints `STOP POD BEFORE ANALYSIS` when it
returns; stopping the cloud pod itself remains an operator action.

After reviewing the artefacts, prepare a JSON mapping from each bounded
`mode/point/length_class` key to its human-approved physical justification,
then materialise a strict `tolerances.json` locally:

```bash
python scripts/step2_promote_calibration.py \
  --run-dir artifacts/step2/runs/a40-YYYY-MM-DD \
  --justifications path/to/justifications.json
```

Promotion writes neither an ADR, governance record, official verdict nor Git
commit; those remain explicit human-reviewed actions.

Model loading requires several dozen gigabytes of memory and may take several
minutes. Decisive measurements must use the checkpoint, configuration, and
environment documented in the reports.

## Repository Layout

```text
formic/
├── formic/                main Python package
│   ├── backbone/          checkpoint, groups, boundaries, runner
│   ├── config/            configuration schema and loader
│   ├── science/           determinism, identity, environment, registry
│   ├── runtime/           transaction engine, planned
│   ├── contracts/         ContractIR and compiler, planned
│   ├── state/             State Fabric, planned
│   ├── actions/           typed actions, planned
│   ├── validation/        monitor, sandbox, and validation, planned
│   ├── eval/              evaluation and baselines, planned
│   ├── episodes/          episode production, planned
│   └── sidecars/          neural components, planned
├── configs/               YAML configurations and frozen prompts
├── docs/adr/              architecture decision records
├── tests/                 tests, weight-free by default
├── scripts/               acceptance and experiment tools
├── experiments/           append-only experiment registry
├── reports/               development and measurement reports
├── artifacts/             execution outputs, not versioned
├── PROJECT.md             complete project context
└── STATUS.md              current status board
```

## Contribution Rules

Before making a significant change, read [`docs/conventions.md`](docs/conventions.md).
The essential principles are:

- the checkpoint audit is the primary technical authority;
- project steps are strictly ordered;
- architectural decisions must not be invented implicitly;
- every new behavior is behind a configuration flag disabled by default;
- the all-disabled mode must preserve Qwen behavior;
- every measurement must be reproducible and retain its metadata;
- Qwen cells must not be rewritten;
- audit constraints A1 through A12 must be respected;
- no later step may begin before the previous step has been validated.

## Main Documentation

- [`PROJECT.md`](PROJECT.md): context, motivation, audit, CAPE-R architecture,
  and the complete implementation plan;
- [`STATUS.md`](STATUS.md): current brick and step status;
- [`docs/conventions.md`](docs/conventions.md): working rules and A1-A12 audit
  constraints;
- [`docs/adr/`](docs/adr/): architecture decisions;
- [`docs/adr/ADR-0005-identity-protocol-and-tolerances.md`](docs/adr/ADR-0005-identity-protocol-and-tolerances.md):
  validated horizon-8 protocol and tolerance governance;
- [`reports/step2_a40_campaign_cost_plan.md`](reports/step2_a40_campaign_cost_plan.md):
  final A40 sequence, forwards, transfers, and preflight estimates;
- [`reports/`](reports/): technical and step reports;
- [`experiments/REGISTRY.md`](experiments/REGISTRY.md): experiment registry.

## Summary

Uly Code is the agentic system that understands, scopes, and orchestrates
software-engineering tasks. Formic is the execution model designed to transform
those scoped tasks into precise, validated, and verifiable software actions.

The project intentionally starts with a strict foundation: preserve the behavior
of the base Qwen model, measure the runtime, build typed state and transactions,
and then add specialized capabilities incrementally.

Formic will not be judged only by the quality of its text. Its success will be
measured by whether it can execute long, complex, well-scoped tasks with the
reliability and code quality of frontier-scale systems, while requiring far less
inference cost when paired with a frontier orchestration model in Uly Code.
