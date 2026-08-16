# Formic

A software-engineering **executor** model built on the Qwen3.8-27B checkpoint as
a pretrained neural substrate.

Formic is not "Qwen plus a prompt". The target architecture (CAPE-R,
`FINAL_TARGET_ARCHITECTURE.md`) changes the unit of model operation from an
open-ended conversation to a **bounded execution transaction**: a protected
instruction contract, a versioned typed state, selected repository evidence, one
depth-bounded neural decision, one typed action, deterministic validation, and an
atomic commit or rejection.

This repository is **part 1**: the foundations. The goal here is not yet to beat
Qwen3.8-27B — it is to obtain a system in which the checkpoint is integrated
cleanly, its original full-depth path stays intact and continuously verified, the
16 hybrid groups are explicitly controllable as a *view*, and the project's
scientific tooling exists from day one.

## Current status

See [`STATUS.md`](STATUS.md). Step 1 (repository skeleton + backbone
integration) is implemented; steps 2–8 are pending, in strict order.

## Quick start

```bash
export PYTHONPATH=$PWD

# weight-free (seconds)
python -m formic.cli verify                 # structural verification, CI entry point
python -m formic.cli structure              # the 16 hybrid groups, 17 boundaries
python -m formic.cli inventory              # strict tensor inventory (headers only)
python -m formic.cli config                 # resolved run config + hash
python -m formic.cli env                    # backend/environment record
python -m pytest tests/ -q                  # weight-free test suite

# with weights (~55 GB, several minutes)
python -m formic.cli load                            # strict-load report
python -m formic.cli generate --prompt "..." --chat  # native generation
python scripts/step1_acceptance.py --stage all       # step-1 exit checklist
```

## What exists today

| Module | Role |
|---|---|
| `formic/backbone/constants.py` | Audited facts of the checkpoint (64 layers, 16 groups, state shapes, byte formulas, reserved rows). Single source of truth for every number. |
| `formic/backbone/inventory.py` | Strict tensor inventory (A12). Reads safetensors headers only; declares and counts every intentional exclusion; matches the loaded model back to the checkpoint both ways. |
| `formic/backbone/loader.py` | Loads the stock HF implementation. Text-only mode uses `Qwen3_5ForCausalLM` + a pure key rename so the vision tower is *never constructed* (A7, ADR-0002). |
| `formic/backbone/groups.py` | The 16 hybrid groups as a **view** over intact modules (A11): group ↔ layer mapping, `3 GDN + 1 Full Attention` pattern, 17 boundaries, contiguous-prefix helpers. |
| `formic/backbone/boundaries.py` | The 17 inert insertion points. With nothing enabled, **no hook is registered at all**, so the forward graph is byte-for-byte stock. |
| `formic/backbone/runner.py` | Native generation with pinned thinking/sampling policies; forward pass with logit fingerprints for identity work. |
| `formic/backbone/torch_compat.py` | Environment shim for torch 2.4 × transformers 5.8 (ADR-0003). Touches no Qwen code. |
| `formic/config/` | Strict run-config schema. Unknown keys are fatal; every Formic mechanism is a flag defaulting to OFF; part-2 flags are refused. |
| `formic/science/` | Experiment registry (`EXP-…`) and environment pinning. |

## The invariant

```text
all flags OFF  ==  Qwen3.8-27B
```

Today this is enforced structurally (no hooks registered, no module added, strict
inventory, stock classes) and checked preliminarily by
`scripts/step1_acceptance.py`. From step 2 it becomes a **blocking CI identity
check**: any break freezes development until it is fixed.

## Ground rules

Read [`docs/conventions.md`](docs/conventions.md) before contributing. The short
version: the audit outranks the architecture, which outranks the plan, which
outranks this repo; steps run in strict order; the agent never decides
architecture; every new behaviour ships behind an OFF flag; every quoted number
carries a config hash, a commit, seeds and an `EXP-…` id.

## Layout

```text
formic/       python package (backbone, runtime, contracts, state, actions,
              validation, eval, episodes, sidecars, config, science)
configs/      run configs + frozen prompt sets
docs/adr/     architecture decision records
tests/        weight-free by default
scripts/      acceptance / experiment entry points
experiments/  append-only EXP registry
reports/      step reports
artifacts/    run outputs (git-ignored)
```
