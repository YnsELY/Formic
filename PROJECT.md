# Formic — Project Context

**Read this before touching any code.** This document exists so that any AI
agent (or human) can pick up this project with zero prior conversation history
and understand what it is, why it exists, and how the pieces fit together. It
describes the *project*, not the current implementation status — for status,
see `STATUS.md` and `reports/`.

---

## 1. What this project is, in one paragraph

Formic is a **software-engineering executor model**: a neural architecture
derived from the pretrained weights of a large checkpoint called
**Qwen3.8-27B**, redesigned to be exceptional at *executing* well-specified
coding tasks (as opposed to being a general-purpose conversational or reasoning
model). It sits underneath a more powerful "orchestrator" model in an agentic
system: the orchestrator investigates, diagnoses, and scopes a problem; Formic
receives an already-framed task — objective, constraints, context — and must
execute it with high fidelity, correctness, reliability, and efficiency, while
avoiding the failure modes (overthinking, scope drift, premature or missed
completion) that plague current coding agents.

The project does **not** pretrain a model from scratch. It **transplants**
selected weights from the Qwen3.8-27B checkpoint into a new execution
architecture, preserving only what is necessary to keep those weights useful,
and building new structure (a transactional execution engine, typed
instructions, typed code actions, explicit long-horizon state) around them.

---

## 2. Why this project exists

### 2.1 The problem with "just use an agent harness around a strong model"

Modern coding agents wrap a strong LLM in a prompt/tool loop. This works, but
inherits structural weaknesses of the underlying decoder:

- **Instructions dilute.** System/user instructions live in the same token
  stream as untrusted tool output and repository text; over a long task, early
  constraints can be diluted, forgotten, or overridden by later content
  (including adversarial content in files, comments, or logs).
- **Progress is implicit.** "What has been done, what remains, what failed
  before" lives in conversational history or ad-hoc summaries, not in a typed,
  recoverable state. Long tasks degrade as context grows or gets compacted.
- **Reasoning is unbounded and undirected.** Thinking models (including this
  one) are prone to *overthinking*: over-exploration (considering solutions
  already known to be sufficient) and over-verification (re-checking settled
  facts) — burning tokens and latency without improving correctness, and
  sometimes actively causing scope creep (the model "fixes" more than asked).
- **Output is free-form text.** Code edits are generated as raw text/diffs with
  no structural guarantee that the target still matches what was read, no
  enforced scope boundary, and no separation between "I produced an edit" and
  "the task is complete."
- **Completion is a language act.** The model decides it's done by *saying* so,
  which is not evidence of anything.

### 2.2 Why Qwen3.8-27B specifically

Community and internal research (captured in the audit, see §4) shows this
checkpoint is an unusually strong substrate for this purpose: a ~27B-parameter
model with a modern hybrid architecture (linear-attention/Gated-DeltaNet layers
interleaved with full attention), strong code and agentic benchmark results,
large native context (262,144 tokens), and a trained Multi-Token-Prediction
(MTP) module. Its main documented weakness for our purposes is not a lack of
capability but *misallocated* capability: it reasons well but doesn't always
know when to stop reasoning and act. The project's thesis is:

> Don't try to make the model reason *more*. Make it reason *better allocated*,
> and wrap it in a system that makes execution — not conversation — the native
> unit of work.

### 2.3 The strategic approach

```text
1. Audit the checkpoint exhaustively — understand its actual physical/runtime
   behavior, not assumptions about it (done; see §4).
2. Design a target architecture from first principles for the executor
   objective, then determine which parts of the checkpoint's weights can be
   transplanted into it, preserving only what keeps them useful (done; see §5).
3. Have independent reviewers stress-test that architecture against the audit,
   correct its flaws, and merge into a final target spec (done; see §5).
4. Implement incrementally, in two parts:
   Part 1 (in progress) — build the foundations: clean integration of the
     checkpoint, a transactional execution engine, typed actions, evaluation
     harness, and the first neural sidecars — all while proving continuously
     that nothing about the original model's behavior has been broken.
   Part 2 (not started) — the full CAPE-R architecture: progressive-depth
     inference, protected instruction memory, speculative decoding, learned
     routing, reinforcement learning.
```

---

## 3. The system this model will live inside

```text
FRONTIER MODEL / ORCHESTRATOR
        |
        v
  understand the problem
  investigate
  strategize, plan globally
  decompose into precise tasks
        |
        v
   precise, scoped task
        |
        v
  FORMIC (this project) — the executor
        |
        v
  execute: read, edit, verify, correct, complete
```

Formic is **not** expected to do open-ended investigation or high-level
strategy — that's the orchestrator's job. Formic receives a task that is
already diagnosed, scoped, and constrained, and must be exceptional at turning
it into a correct, minimal, verified change. This division of labor is why
Formic's design leans hard into *discipline* (typed contracts, typed actions,
deterministic completion, bounded recovery) rather than into open-ended
intelligence.

### 3.1 Qualities Formic must maximize

Instruction fidelity, code correctness, code quality, reliability, long-horizon
execution, scope adherence, controllability, efficiency, low latency,
completion awareness, minimal unnecessary changes, minimal unnecessary
reasoning.

### 3.2 What "executor" does *not* mean

It does not mean reasoning-free. Formic must still understand a task deeply,
understand *why* it's being asked, analyze the consequences of a change, plan
locally, reason during a long task, verify its own work, detect that it's going
in a wrong direction, question and revise a decision, correct errors locally,
decide whether more thought is needed, and decide when a task is genuinely
done. The distinction the whole project is built around:

```text
FRONTIER ORCHESTRATOR        EXECUTOR (Formic)
  open-ended reasoning         task-understanding reasoning
  investigation                implementation reasoning
  global strategy               constraint reasoning
  problem discovery             verification reasoning, correction

REASONING MUST SERVE EXECUTION.
```

---

## 4. The checkpoint audit — ground truth about the substrate

Before any architecture was designed, the checkpoint was exhaustively,
non-destructively audited. **The audit is the highest authority in this
project**: nothing may be assumed compatible with the checkpoint unless the
audit confirms it, and any conflict between later documents and the audit is
resolved in the audit's favor.

Location: `/workspace/audits/qwen3_8_27b/` — 15 numbered reports plus a
`FINAL_AUDIT_REPORT.md` synthesis, raw JSON results, logs, and the scripts that
produced them. Read `FINAL_AUDIT_REPORT.md` first for the synthesis, then the
numbered reports for depth on any topic.

### 4.1 The central identity finding

The product is named **Qwen3.8-27B**, but its runtime architecture is
**Qwen3.5** (`Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`, running
through `transformers==5.8.0`'s `modeling_qwen3_5.py`). There is no distinct
"Qwen3.8" implementation class — this is documented by Qwen as intentional
(Qwen3.8 is built on the Qwen3.5 architectural foundation), not a checkpoint
error.

### 4.2 Structural facts (all directly measured, not assumed)

- **64 decoder layers**, hidden size 5,120, FFN intermediate 17,408, vocabulary
  248,320, native context 262,144 tokens, untied embedding/LM-head.
- **16 hybrid groups**, each exactly `[GatedDeltaNet, GatedDeltaNet,
  GatedDeltaNet, Full Attention]` — full-attention layers sit at indices 3, 7,
  11, ..., 63 (0-indexed). This pattern was hypothesized and then independently
  verified against config, instantiated modules, and weight names.
- **27,781,427,952** total BF16 parameters stored. Of these, **424,699,392**
  belong to a trained **Multi-Token-Prediction (MTP)** module that the
  standard Transformers runtime silently *ignores* on load
  (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) — the weights exist and
  are usable, but nothing in stock `transformers` currently executes them.
  Another ~460.7M parameters belong to an optional vision tower.
- **Gated DeltaNet (GDN) layers** carry two persistent states per layer: a
  convolutional state (`[B, 10240, 4]`) and a recurrent delta-rule state
  (`[B, 48, 128, 128]`). The recurrence computes in FP32 but the *persisted*
  state is BF16 — meaning re-segmenting a sequence differently (e.g. monolithic
  prefill vs. chunked continuation) produces slightly different numbers at the
  bit level, even though the math is "the same." This has real consequences for
  any caching/replay strategy.
- **GDN state mutation is in-place and non-idempotent.** Replaying the same
  tokens through a GDN layer a second time re-applies decay and rewrites state
  — it does *not* reproduce the first pass. `use_cache=False` does **not**
  make a forward pass read-only if a cache object is supplied.
- **Attention KV cache** grows by reallocation (`cat`), not in-place mutation;
  replaying a segment duplicates KV entries and desynchronizes the causal mask.
- Both cache types combine in one heterogeneous `DynamicCache` object; there is
  no dedicated cache class for this architecture. Full snapshot/restore/fork of
  this hybrid cache is *possible* (validated experimentally, via deep-cloning)
  but expensive: full-context (262k token) snapshots run to ~16 GiB.
- **RMSNorm has two different conventions** in the same network: the text-side
  norm is zero-centered (`(1+weight)·RMS(x)`), while the GDN-internal gated
  norm uses the standard convention (`weight·RMS(x)·silu(z)`). Conflating them
  silently breaks the model.
- The chat template ships tool-use support and a **thinking mode** with an
  `enable_thinking` switch that **defaults to true** — this is a thinking-first
  model whose agentic benchmark scores were produced with visible deliberation
  enabled.
- 243 embedding/LM-head rows (IDs 248,077–248,319) exist in the checkpoint with
  **no tokenizer mapping** — reserved, untrained capacity that can eventually
  host new control tokens, but is meaningless to a model that hasn't been
  trained on them.

### 4.3 Why this matters for design

Every one of these facts directly shaped what the target architecture is and
is not allowed to do: e.g., because GDN replay is non-idempotent and expensive
to snapshot at scale, the architecture avoids any design that requires
replaying or branching neural cache state, and instead puts durable state in an
external, typed system. Because the model is thinking-default, the architecture
does not strip out visible deliberation — it disciplines it. Because full and
partial attention/GDN layers alternate in a fixed, audited pattern, the
architecture treats a "hybrid group" (one full pattern repetition) as the
smallest safe unit to reason about architecturally, never a single layer in
isolation and never an arbitrary layer subset.

---

## 5. The target architecture — CAPE-R

### 5.1 Documents and provenance

The target architecture went through three stages, all preserved for
provenance:

1. `/workspace/target_architecture_research/ALL_TARGET_ARCHITECTURE.md` — the
   original architecture proposal (codename **CAPE-27B**, "Contract-Aware
   Progressive Executor"), designed from the audit and the project's
   requirements.
2. Two **independent adversarial reviews** of that proposal, each checking
   every claim against the audit, scoring the architecture, and proposing
   corrections and new mechanisms:
   `/workspace/architecture_reviews/INDEPENDENT_ARCHITECTURE_REVIEW.md` and
   `INDEPENDENT_ARCHITECTURE_REVIEW_K3.md`. Both reviews, produced
   independently, converged on the same verdict (architecture is fundamentally
   sound, four specific corrections needed) — which is itself strong evidence
   the corrections are real.
3. **`/workspace/target_architecture_research/FINAL_TARGET_ARCHITECTURE.md`**
   — the merged, corrected, final target specification, codenamed **CAPE-R**
   (Contract-Aware Progressive Executor, Revised). **This is the architecture
   Formic is being built toward.** It supersedes the original CAPE-27B
   document; the original and both reviews remain as historical record.

**Read `FINAL_TARGET_ARCHITECTURE.md` in full before doing architectural work.**
What follows here is a summary sufficient to orient, not a replacement.

### 5.2 The core idea: transactions, not conversation

CAPE-R changes the fundamental unit of model operation from an open-ended
token sequence to a **bounded execution transaction**:

```text
trusted contract + versioned state + selected repository evidence
    -> one depth-bounded neural decision
    -> one typed action / state proposal
    -> deterministic validation and external evidence
    -> atomic commit or rejection
    -> next canonical state version
```

A long task is a *sequence* of these transactions, not one long decode. Each
transaction: reads an immutable, versioned instruction contract; reads a typed,
external state (not free-form context); proposes exactly one principal
action (an edit, a tool call, a question, a state update, or completion);
gets that action deterministically validated and applied in a sandbox; and
either commits atomically or is rejected without side effects. Neural caches
(GDN state, KV cache) are treated as disposable per-transaction scratch, never
as the source of truth for task progress — durable state lives externally, in
a typed, versioned "State Fabric."

### 5.3 The four architectural planes

```text
Control Plane   — immutable, versioned ContractIR (instructions compiled from
                  authoritative sources); a non-neural Reference Monitor that
                  enforces authority/scope and can reject actions but never
                  invents one.
State Plane     — repository snapshot, artifact graph (files/symbols/tests),
                  task DAG of obligations, evidence bank, append-only ledger.
                  This is where "what has been done, what remains, what
                  failed" actually lives — not in conversational history.
Execution Plane — the preserved Qwen trunk, run as a contiguous prefix of its
                  original 64 layers (never reordered, never replayed, never
                  branched at the neural-cache level), producing typed outputs
                  via pointer heads and a constrained output grammar instead of
                  free-form text.
Commit Plane    — schema/scope/hash validation, sandboxed application,
                  required checks (tests, parsers, type checks), then atomic
                  commit or reject. Completion is a *deterministic predicate*
                  over verified evidence — the model can propose it's done, it
                  cannot declare it.
```

### 5.4 Adaptive compute (how "don't overthink" becomes structural)

Rather than a single fixed forward pass, CAPE-R selects how much of the network
to run per transaction, using calibrated exit points at natural architectural
boundaries (after 8, 12, or all 16 hybrid groups — never mid-group, never an
arbitrary layer): routine work exits early and cheaply; harder, riskier, or
previously-failed work runs the full stack. Depth is decided *during prefill*
(at the point of maximum available information, at zero wasted compute versus
routing eagerly) and is **frozen once decoding starts** — the architecture
never asks a skipped layer to catch up on tokens it never saw. Visible
deliberation (the model's native `<think>` capability) is kept but
**budgeted per depth level** and trained to shrink via preference optimization,
rather than removed outright — because this is a thinking-default model and
stripping the channel entirely risks destroying the very capability the
project depends on. A handful of structural anti-overthinking guards do the
rest: one action per transaction, no generic "reflect" action, reconsideration
gated on new external evidence (never self-generated doubt), duplicate-attempt
detection, and a bounded, novelty-gated repair budget after failure (not
open-ended retry).

### 5.5 Weight reuse discipline

Every Qwen tensor that is kept is kept **exactly**: same shape, same equations,
same residual/normalization ordering, no re-purposing, no silent reinterpretation.
New capability is added *around* the trunk — small additive, zero-initialized
modules (role/trust embeddings on input tokens, low-rank exit bridges, typed
output heads, a handful of learned "decision slot" positions used as a compact
workspace) — never by modifying a pretrained cell's internals. A hard
architectural invariant, checked continuously: with every new mechanism turned
off, the system must reproduce the original Qwen3.8-27B checkpoint's behavior
exactly. This invariant is what makes it possible to add capability without
ever silently degrading what the checkpoint already does well.

### 5.6 What the architecture deliberately does *not* do

No recurrent reuse of the trunk's groups (they were never trained as a shared
transition function), no mixture-of-experts-style routing over groups, no
neural-cache branching used as a planning/backtracking mechanism (the audit
shows this is expensive and fragile — branching happens at the *state* level,
which is cheap, not the neural-cache level, which is not), no fully free-form
code generation for structural edits (edits are typed and pointer-bound, though
raw payload text generation is preserved for actual code content), and no
claim of architectural success without matched-baseline, controlled evaluation.

---

## 6. The implementation plan — how CAPE-R becomes Formic

Location:
`/workspace/docs/implementation/formic_plan_implementation_initial.md`.
This plan governs *how* the architecture above gets built, in eight strictly
ordered steps, split into two parts.

### 6.1 Part 1 — foundations (steps 1–8)

The explicit goal of part 1 is **not** yet to make Formic better than
Qwen3.8-27B. It's to get to a system where: the checkpoint is integrated
cleanly with zero behavioral change; the original full-depth path stays
intact, measurable, and continuously verified; the 16 hybrid groups are
explicitly controllable as a view (not a rewrite); Formic has its transactional
engine (contract, state, monitor, atomic commit); the model can propose
structured changes that get tested, validated, committed, or rejected; small,
well-scoped software tasks are executed end-to-end; **baseline numbers exist**
— pinned baselines, a frozen evaluation suite, and a measured answer to "does
the runtime alone help?"; training-data production is industrialized; and the
first neural sidecars are trained, ablated, and validated.

The eight steps, in strict order (each step's exit checklist must be fully
green and human-validated before the next begins):

1. **Formic foundation + backbone integration** — repository skeleton,
   scientific tooling (experiment registry, ADRs, config schema), and Qwen3.8-27B
   loaded as an unmodified substrate with the 16 hybrid groups exposed as an
   explicit, testable view.
2. **Identity baseline + numeric tolerances + snapshot/restore** — prove, in a
   single blocking command, that Formic with everything off reproduces
   Qwen3.8-27B exactly (within measured tolerances), and build the
   snapshot/restore/fork primitive needed everywhere later.
3. **Evaluation harness + pinned baselines** — frozen test suites, metrics,
   ≥3-seed baselines, and the first calibration decisions (thinking
   configuration, sampling temperature) made on data.
4. **Transactional engine, full-depth** — ContractIR + its compiler, the State
   Fabric, the transaction engine, and the Reference Monitor — with the neural
   network still running unmodified full-depth Qwen.
5. **Software-native action interface** — typed actions (edits, tool calls,
   state updates, completion, abstention) instead of free-form text, with
   validation, sandboxing, and atomic application.
6. **First end-to-end loop + FORMIC-M0** — small, scoped software tasks run
   completely through the system, and the first real measurement is taken:
   does the transactional runtime *by itself* (with zero new neural weights)
   improve on raw Qwen3.8-27B under an agent harness? This is baseline #2 that
   every future neural addition must beat.
7. **Episode production machine** — industrialize the generation of labeled
   execution episodes (the runtime itself becomes the labeling machine) for
   the training that starts in step 8.
8. **First L2 (full-depth) neural sidecars + FORMIC-M1** — train the first
   small, additive neural components (contract/state understanding, pointer
   heads, completion signals) with the Qwen trunk frozen, and validate each
   one earns its place by beating FORMIC-M0 in isolated ablation.

### 6.2 Part 2 — not started, deliberately deferred

Everything that makes CAPE-R "progressive" and speculative is explicitly out of
scope until FORMIC-M1 is validated: shallow-exit bridges (L0/L1), route-specific
low-rank adapters, learned depth routing, tight scratch/thinking budgets, a
learned "continue reasoning or act now" head, cross-transaction prefix caching,
depth-speculative decoding, the full GDN state-rollback protocol needed for any
speculative decoding, MTP integration, long-horizon (50–500 transaction)
training, outcome/preference optimization, reinforcement learning, multi-GPU
serving, and the vision path. These aren't forgotten — they're CAPE-R's actual
differentiators — they're sequenced *after* a solid, measured foundation
exists, per the plan's strict-ordering rule.

### 6.3 Non-negotiable rules governing implementation

These apply to every step without exception and are the rules an agent working
on this repo must internalize:

1. Steps execute in strict order; a step does not start until the previous
   step's exit checklist is fully satisfied and a human has signed off.
2. **The implementing agent never decides architecture.** No unspecified
   mechanism, no silently modified threshold, no "equivalent" shortcut.
   Genuine ambiguity is raised as a precise question, not resolved
   unilaterally, and structuring decisions that *are* made get an ADR.
3. Every new behavior lives behind a config flag, **off by default**. The
   "everything off" configuration must always reproduce Qwen3.8-27B exactly —
   this is checked continuously, not just claimed.
4. Every step produces: a dedicated branch, tests, a short step report (what
   was done, what was measured, deviations from plan), an updated status
   board, and ADRs for structuring decisions.
5. A step begins with a detailed technical breakdown (files, classes,
   sub-task order) proposed and validated before implementation starts.
6. Part-1 scope constraints: text only (vision path bypassed in code), Python
   only for target repositories under test, batch size 1, one candidate per
   transaction, no training before step 8. BF16 is the reference for any
   decisive measurement; quantized formats may be used to iterate but never to
   close an exit checklist.

A registry of twelve audit-derived constraints (A1–A12 — e.g. "never assume
`use_cache=False` makes a forward pass read-only," "never construct a cache
object without the model's config," "the two RMSNorm conventions must never be
unified," "a text-only build must literally not construct the vision tower in
code") must be re-read at the start of every step and is cited wherever it
applies. These exist because each one maps to a specific, verified way the
audit found this checkpoint's runtime could be silently misused.

---

## 7. Repository orientation

The implementation lives at `/workspace/formic`. Package layout (see
`docs/adr/ADR-0001-repository-conventions.md` for the full rationale):

```text
formic/
├── formic/                # the importable python package
│   ├── backbone/          # checkpoint integration, hybrid-group view,
│   │                      #   boundary hooks, generation runner (steps 1-2)
│   ├── config/            # strict run-config schema + loader
│   ├── science/           # experiment registry, environment/determinism pinning
│   ├── runtime/           # transaction engine                    (step 4)
│   ├── contracts/         # ContractIR + its compiler              (step 4)
│   ├── state/             # State Fabric                           (step 4)
│   ├── actions/           # typed action grammar, IRs, lowering    (step 5)
│   ├── validation/        # reference monitor, sandbox, checks     (4-5)
│   ├── eval/               # evaluation harness, suites, baselines  (step 3)
│   ├── episodes/          # episode production machine             (step 7)
│   ├── sidecars/          # neural components                      (step 8)
│   └── cli.py
├── configs/                # run configs (YAML), frozen prompt sets
├── docs/
│   ├── adr/                # architecture decision records
│   ├── conventions.md      # working rules, audit-constraint registry, policies
│   └── implementation/     # the plan document itself
├── tests/                  # weight-free by default (`weights` marker for the rest)
├── scripts/                 # acceptance / experiment entry points
├── experiments/             # append-only experiment registry
├── reports/                 # step reports (committed)
├── artifacts/               # run outputs (git-ignored)
└── STATUS.md                # per-brick status board
```

Related repositories/directories outside `formic/` that any agent should know
about:

```text
/workspace/Qwen3.8-27B/                        the checkpoint itself
/workspace/audits/qwen3_8_27b/                  the full technical audit (§4)
/workspace/target_architecture_research/        CAPE-27B and CAPE-R architecture docs (§5)
/workspace/architecture_reviews/                the two independent adversarial reviews
/workspace/docs/implementation/                 the implementation plan (§6)
```

**Order of authority when documents conflict:** the checkpoint audit outranks
`FINAL_TARGET_ARCHITECTURE.md`, which outranks the implementation plan, which
outranks anything in this repository. If code, a report, or a prior decision
contradicts a higher-authority document, the higher-authority document wins and
the discrepancy should be raised, not silently resolved.

For where the project currently stands (which steps are done, what's measured,
what's pending human sign-off), read `STATUS.md` and the latest file in
`reports/` — this document deliberately does not track that, so that it stays
accurate as a description of the project regardless of implementation
progress.
