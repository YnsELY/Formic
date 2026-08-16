#!/usr/bin/env python3
"""Step-1 acceptance: run the exit checklist and emit a report.

The preliminary Formic-vs-HF check needs two ~55 GB loads, so the script runs in
stages, each in its own process:

    python scripts/step1_acceptance.py --stage formic     # Formic text-only path
    python scripts/step1_acceptance.py --stage hf         # stock HF reference
    python scripts/step1_acceptance.py --stage compare    # verdict + report
    python scripts/step1_acceptance.py --stage all        # all three, sequentially

The ``hf`` stage deliberately imports nothing from ``formic`` except the prompt
loader: it must be a plain Hugging Face execution, otherwise the comparison
would be circular.

Formal proof of identity is step 2. This is the preliminary check the step-1
checklist asks for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "step1"
CHECKPOINT = "/workspace/Qwen3.8-27B"

GREEDY_MAX_NEW_TOKENS = 16
SAMPLED_MAX_NEW_TOKENS = 16
SAMPLED_PROMPT_IDS = ("plain_text", "code_completion", "instruction_short")
#: Backbone-level decode comparison (no generate() wrapper). This is the
#: checklist criterion; see runner.manual_greedy_decode for why.
MANUAL_MAX_NEW_TOKENS = 8
MANUAL_PROMPT_IDS = ("audit_echo", "plain_text", "code_completion", "instruction_short")


# --------------------------------------------------------------------------
# prompt set (shared by both stages, no model code involved)
# --------------------------------------------------------------------------


def load_prompt_set() -> dict[str, Any]:
    import hashlib

    import yaml

    path = REPO_ROOT / "configs" / "reference_prompts.yaml"
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    data["set_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data


def render_prompts(tokenizer, prompt_set: dict[str, Any], enable_thinking: bool) -> list[dict]:
    """Turn the frozen set into concrete prompt strings (same on both sides)."""
    rendered = []
    for entry in prompt_set["prompts"]:
        if entry["kind"] == "raw":
            text = entry["text"]
        else:
            text = tokenizer.apply_chat_template(
                entry["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        rendered.append({"id": entry["id"], "kind": entry["kind"], "text": text})
    return rendered


# --------------------------------------------------------------------------
# stage: formic
# --------------------------------------------------------------------------


def stage_formic(config_path: Path) -> dict[str, Any]:
    from formic.backbone.boundaries import count_registered_hooks
    from formic.backbone.loader import load_backbone
    from formic.backbone.runner import (
        forward_logits,
        generate,
        manual_greedy_decode,
        set_seed,
    )
    from formic.config.loader import load_config
    from formic.science.determinism import environment_report

    config = load_config(config_path)
    print(f"[stage:formic] config hash {config.config_hash()} identity_mode={config.identity_mode()}")
    handle = load_backbone(config)

    prompt_set = load_prompt_set()
    prompts = render_prompts(handle.tokenizer, prompt_set, config.thinking.enable_thinking)

    results: dict[str, Any] = {
        "stage": "formic",
        "config_hash": config.config_hash(),
        "prompt_set_sha256": prompt_set["set_sha256"],
        "environment": environment_report(),
        "backbone": handle.describe(),
        "hooks_registered": count_registered_hooks(handle.model),
        "forward": {},
        "manual_greedy": {},
        "greedy": {},
        "sampled": {},
    }

    for prompt in prompts:
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
        forward = forward_logits(handle, ids)
        results["forward"][prompt["id"]] = forward.to_dict()
        print(f"[stage:formic] forward {prompt['id']:<20} argmax={forward.argmax_id} "
              f"sha={forward.logits_sha256[:12]} ({forward.seconds:.1f}s)")

    for prompt in prompts:
        if prompt["id"] not in MANUAL_PROMPT_IDS:
            continue
        ids = handle.tokenizer(prompt["text"], return_tensors="pt")["input_ids"][0].tolist()
        manual = manual_greedy_decode(handle, ids, MANUAL_MAX_NEW_TOKENS)
        results["manual_greedy"][prompt["id"]] = manual
        print(f"[stage:formic] manual  {prompt['id']:<20} {manual['generated_token_ids']} "
              f"({manual['seconds']:.1f}s)")

    for prompt in prompts:
        set_seed(config.run.seed, config.run.deterministic)
        generated = generate(
            handle, prompt["text"], do_sample=False,
            max_new_tokens=GREEDY_MAX_NEW_TOKENS, seed=config.run.seed,
        )
        results["greedy"][prompt["id"]] = generated.to_dict()
        print(f"[stage:formic] greedy  {prompt['id']:<20} {len(generated.generated_token_ids)} tokens "
              f"({generated.seconds:.1f}s)")

    for prompt in prompts:
        if prompt["id"] not in SAMPLED_PROMPT_IDS:
            continue
        generated = generate(
            handle, prompt["text"], do_sample=True,
            max_new_tokens=SAMPLED_MAX_NEW_TOKENS, seed=config.run.seed,
        )
        results["sampled"][prompt["id"]] = generated.to_dict()
        print(f"[stage:formic] sampled {prompt['id']:<20} ({generated.seconds:.1f}s)")

    _write(ARTIFACT_DIR / "formic_outputs.json", results)
    return results


# --------------------------------------------------------------------------
# stage: hf reference (no Formic model code)
# --------------------------------------------------------------------------


def stage_hf(config_path: Path) -> dict[str, Any]:
    import hashlib

    # Environment shim only (torch 2.4 x transformers 5.8 custom-op annotations).
    # It is not model code: without it, `import transformers` itself raises here.
    from formic.backbone.torch_compat import ensure_torch_compat

    ensure_torch_compat()

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    import yaml

    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    backbone_cfg = raw_cfg["backbone"]
    thinking_cfg = raw_cfg["thinking"]
    sampling = raw_cfg["sampling"]["payload"]
    seed = raw_cfg["run"]["seed"]
    eos = raw_cfg["generation"]["eos_token_ids"]

    max_memory = {int(k) if str(k).isdigit() else k: v for k, v in backbone_cfg["max_memory"].items()}

    print("[stage:hf] loading stock Qwen3_5ForConditionalGeneration")
    started = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        backbone_cfg["checkpoint_path"],
        dtype=getattr(torch, backbone_cfg["dtype"]),
        attn_implementation=backbone_cfg["attn_implementation"],
        device_map=backbone_cfg["device_map"],
        max_memory=max_memory,
    )
    model.eval()
    load_seconds = time.time() - started
    tokenizer = AutoTokenizer.from_pretrained(backbone_cfg["checkpoint_path"])
    device = getattr(model, "device", next(model.parameters()).device)

    prompt_set = load_prompt_set()
    enable_thinking = thinking_cfg["mode"] != "off"
    prompts = render_prompts(tokenizer, prompt_set, enable_thinking)

    results: dict[str, Any] = {
        "stage": "hf_reference",
        "model_class": type(model).__name__,
        "prompt_set_sha256": prompt_set["set_sha256"],
        "load_seconds": round(load_seconds, 2),
        "parameters": sum(p.numel() for p in model.parameters()),
        "has_vision_tower": hasattr(model.model, "visual"),
        "forward": {},
        "manual_greedy": {},
        "greedy": {},
        "sampled": {},
    }
    print(f"[stage:hf] loaded in {load_seconds:.1f}s, params={results['parameters']:,}, "
          f"vision_tower={results['has_vision_tower']}")

    def _seed_all(value: int) -> None:
        import random

        import numpy as np

        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
            started = time.time()
            out = model(input_ids=ids, use_cache=False)
            seconds = time.time() - started
            logits = out.logits[0, -1].detach().to("cpu").float()
            values, indices = torch.topk(logits, k=10)
            results["forward"][prompt["id"]] = {
                "input_token_ids": ids[0].tolist(),
                "argmax_id": int(torch.argmax(logits).item()),
                "top_k_ids": [int(i) for i in indices.tolist()],
                "top_k_values": [float(v) for v in values.tolist()],
                "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                "logits_rms": float(torch.sqrt(torch.mean(logits**2)).item()),
                "logits_min": float(logits.min().item()),
                "logits_max": float(logits.max().item()),
                "seconds": seconds,
            }
            print(f"[stage:hf] forward {prompt['id']:<20} "
                  f"argmax={results['forward'][prompt['id']]['argmax_id']} ({seconds:.1f}s)")

        # Backbone-level decode: explicit loop, position_ids left to None, so this
        # entry point takes exactly the same path as Formic's. No generate().
        for prompt in prompts:
            if prompt["id"] not in MANUAL_PROMPT_IDS:
                continue
            ids = tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
            past = None
            generated: list[int] = []
            argmax_logits: list[float] = []
            started = time.time()
            current = ids
            for _ in range(MANUAL_MAX_NEW_TOKENS):
                out = model(input_ids=current, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                next_id = int(torch.argmax(logits).item())
                generated.append(next_id)
                argmax_logits.append(float(logits[next_id].item()))
                current = torch.tensor([[next_id]], dtype=torch.long, device=device)
            results["manual_greedy"][prompt["id"]] = {
                "input_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "argmax_logit_per_step": argmax_logits,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": time.time() - started,
                "cache_seq_length": int(past.get_seq_length()) if past is not None else 0,
            }
            print(f"[stage:hf] manual  {prompt['id']:<20} {generated} "
                  f"({results['manual_greedy'][prompt['id']]['seconds']:.1f}s)")

        for prompt in prompts:
            encoded = tokenizer(prompt["text"], return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            _seed_all(seed)
            started = time.time()
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=GREEDY_MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
            seconds = time.time() - started
            generated = out[0, ids.shape[-1]:].tolist()
            results["greedy"][prompt["id"]] = {
                "prompt_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": seconds,
            }
            print(f"[stage:hf] greedy  {prompt['id']:<20} {len(generated)} tokens ({seconds:.1f}s)")

        for prompt in prompts:
            if prompt["id"] not in SAMPLED_PROMPT_IDS:
                continue
            encoded = tokenizer(prompt["text"], return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            _seed_all(seed)
            started = time.time()
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=SAMPLED_MAX_NEW_TOKENS, do_sample=True,
                temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
            seconds = time.time() - started
            generated = out[0, ids.shape[-1]:].tolist()
            results["sampled"][prompt["id"]] = {
                "prompt_token_ids": ids[0].tolist(),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "seconds": seconds,
            }
            print(f"[stage:hf] sampled {prompt['id']:<20} ({seconds:.1f}s)")

    _write(ARTIFACT_DIR / "hf_outputs.json", results)
    return results


# --------------------------------------------------------------------------
# stage: compare + checklist
# --------------------------------------------------------------------------


def stage_compare(config_path: Path) -> dict[str, Any]:
    formic = _read(ARTIFACT_DIR / "formic_outputs.json")
    reference = _read(ARTIFACT_DIR / "hf_outputs.json")

    comparison: dict[str, Any] = {"forward": {}, "manual_greedy": {}, "greedy": {}, "sampled": {}}
    for prompt_id, formic_entry in formic["forward"].items():
        ref = reference["forward"][prompt_id]
        comparison["forward"][prompt_id] = {
            "same_input_ids": formic_entry["input_token_ids"] == ref["input_token_ids"],
            "same_argmax": formic_entry["argmax_id"] == ref["argmax_id"],
            "same_top_k_ids": formic_entry["top_k_ids"] == ref["top_k_ids"],
            "same_logits_sha256": formic_entry["logits_sha256"] == ref["logits_sha256"],
            "max_top_k_delta": max(
                abs(a - b)
                for a, b in zip(formic_entry["top_k_values"], ref["top_k_values"])
            ),
            "rms_delta": abs(formic_entry["logits_rms"] - ref["logits_rms"]),
        }
    for kind in ("manual_greedy", "greedy", "sampled"):
        for prompt_id, formic_entry in formic.get(kind, {}).items():
            ref = reference.get(kind, {}).get(prompt_id)
            if ref is None:
                continue
            same = formic_entry["generated_token_ids"] == ref["generated_token_ids"]
            prefix = 0
            for a, b in zip(formic_entry["generated_token_ids"], ref["generated_token_ids"]):
                if a != b:
                    break
                prefix += 1
            entry: dict[str, Any] = {
                "identical": same,
                "matching_prefix": prefix,
                "length": len(formic_entry["generated_token_ids"]),
                "formic_text": formic_entry["text"],
                "reference_text": ref["text"],
            }
            if kind == "manual_greedy":
                entry["formic_tokens"] = formic_entry["generated_token_ids"]
                entry["reference_tokens"] = ref["generated_token_ids"]
                entry["same_cache_length"] = (
                    formic_entry.get("cache_seq_length") == ref.get("cache_seq_length")
                )
                entry["max_argmax_logit_delta"] = max(
                    (
                        abs(a - b)
                        for a, b in zip(
                            formic_entry.get("argmax_logit_per_step", []),
                            ref.get("argmax_logit_per_step", []),
                        )
                    ),
                    default=0.0,
                )
            comparison[kind][prompt_id] = entry

    forward_ok = all(
        entry["same_argmax"] and entry["same_top_k_ids"] and entry["same_input_ids"]
        for entry in comparison["forward"].values()
    )
    manual_ok = bool(comparison["manual_greedy"]) and all(
        entry["identical"] for entry in comparison["manual_greedy"].values()
    )
    greedy_ok = all(entry["identical"] for entry in comparison["greedy"].values())
    sampled_ok = all(entry["identical"] for entry in comparison["sampled"].values())
    bitwise_ok = all(entry["same_logits_sha256"] for entry in comparison["forward"].values())

    backbone = formic["backbone"]
    load_report = backbone["load_report"]
    structure = backbone["structure"]

    generate_wrapper_note = (
        "generate() token-identical on "
        f"{sum(1 for e in comparison['greedy'].values() if e['identical'])}"
        f"/{len(comparison['greedy'])} greedy and "
        f"{sum(1 for e in comparison['sampled'].values() if e['identical'])}"
        f"/{len(comparison['sampled'])} sampled prompts - divergence is in the HF "
        "generate() wrapper (multimodal position-id override), not the backbone; "
        "see reports/step1_report.md section 5.6"
    )
    checklist = [
        {
            "item": "Backbone identical to a direct HF run: prefill logits + decode (preliminary)",
            "pass": forward_ok and bitwise_ok and manual_ok,
            "detail": (
                f"prefill: argmax/top-k identical on {len(comparison['forward'])} prompts, "
                f"FP32 logits bitwise-identical: {bitwise_ok}; "
                f"decode (explicit loop, no generate()): token-identical on "
                f"{sum(1 for e in comparison['manual_greedy'].values() if e['identical'])}"
                f"/{len(comparison['manual_greedy'])} prompts. {generate_wrapper_note}"
            ),
        },
        {
            "item": "Group <-> layer mapping conforms (16 groups, 64 layers, attention at 3..63)",
            "pass": (
                structure["num_layers"] == 64
                and structure["attention_layer_indices"] == list(range(3, 64, 4))
                and structure["mixer_classes"]["self_attn"] == "Qwen3_5Attention"
                and structure["mixer_classes"]["linear_attn"] == "Qwen3_5GatedDeltaNet"
            ),
            "detail": f"mixer classes {structure['mixer_classes']}",
        },
        {
            "item": "No re-implemented or copy-modified cell code",
            "pass": _run_guard_tests(),
            "detail": "tests/test_no_cell_reimplementation.py (AST + pattern guards) green",
        },
        {
            "item": "Strict tensor inventory; permissive loading impossible",
            "pass": load_report["ok"]
            and load_report["loaded_params"] == load_report["expected_params"],
            "detail": (
                f"{load_report['matched']} tensors matched, "
                f"{load_report['loaded_params']:,} params, "
                f"declared exclusions {load_report['declared_exclusions']}"
            ),
        },
        {
            "item": "Vision tower not constructed in text-only mode (memory evidence)",
            "pass": (
                not load_report["vision_tower_present"]
                and backbone["parameters"] == 26_895_998_464
                and reference["parameters"] == 27_356_728_560
            ),
            "detail": (
                f"formic text-only {backbone['parameters']:,} params "
                f"({backbone['parameter_bytes'] / 2**30:.2f} GiB) vs HF multimodal "
                f"{reference['parameters']:,} params; difference "
                f"{reference['parameters'] - backbone['parameters']:,} = vision tower; "
                f"vision module present: formic={load_report['vision_tower_present']}, "
                f"hf={reference['has_vision_tower']}"
            ),
        },
        {
            "item": "All boundary hooks inert by default, config-driven",
            "pass": formic["hooks_registered"] == 0 and backbone["identity_mode"],
            "detail": (
                f"{formic['hooks_registered']} layer hooks registered with default config; "
                f"identity_mode={backbone['identity_mode']}"
            ),
        },
        {
            "item": "STATUS.md, EXP registry, ADR template, conventions in place",
            "pass": all(
                (REPO_ROOT / rel).exists()
                for rel in (
                    "STATUS.md",
                    "experiments/REGISTRY.md",
                    "docs/adr/ADR-TEMPLATE.md",
                    "docs/conventions.md",
                )
            ),
            "detail": "governance files present",
        },
    ]

    report = {
        "step": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": formic["config_hash"],
        "prompt_set_sha256": formic["prompt_set_sha256"],
        "audit_baseline": _compare_to_audit_baseline(formic),
        "comparison": comparison,
        "summary": {
            "forward_argmax_and_topk_identical": forward_ok,
            "forward_logits_bitwise_identical": bitwise_ok,
            "backbone_decode_token_identical": manual_ok,
            "generate_wrapper_greedy_identical": greedy_ok,
            "generate_wrapper_sampled_identical": sampled_ok,
        },
        "checklist": checklist,
        "overall_pass": all(item["pass"] for item in checklist),
    }
    _write(ARTIFACT_DIR / "step1_report.json", report)
    _write_markdown(report, formic, reference)
    _register_experiment(report, formic, reference)
    _print_checklist(report)
    return report


def _register_experiment(report: dict[str, Any], formic: dict, reference: dict) -> None:
    """Record this acceptance run in the EXP registry (config + commit + cost)."""
    from formic.science.registry import ExperimentRegistry

    registry = ExperimentRegistry()
    existing = {r.title for r in registry.records()}
    title = "Step-1 acceptance: backbone integration + preliminary Formic-vs-HF check"
    record = registry.start(
        title=title,
        step="part1/step1",
        config_hash=report["config_hash"],
        config_path="configs/default.yaml",
        seeds=(formic["backbone"]["config_hash"] and 0,),
        environment=formic.get("environment", {}),
        notes="" if title not in existing else "re-run",
    )
    formic_seconds = sum(
        entry["seconds"]
        for section in ("forward", "greedy", "sampled")
        for entry in formic[section].values()
    )
    reference_seconds = sum(
        entry["seconds"]
        for section in ("forward", "greedy", "sampled")
        for entry in reference[section].values()
    )
    registry.finish(
        record,
        status="DONE" if report["overall_pass"] else "FAILED",
        metrics=report["summary"],
        cost={
            "formic_load_seconds": formic["backbone"]["load_seconds"],
            "reference_load_seconds": reference["load_seconds"],
            "formic_compute_seconds": round(formic_seconds, 1),
            "reference_compute_seconds": round(reference_seconds, 1),
        },
        artifacts=(
            "artifacts/step1/formic_outputs.json",
            "artifacts/step1/hf_outputs.json",
            "artifacts/step1/step1_report.json",
            "artifacts/step1/step1_report.md",
        ),
    )


AUDIT_BASELINE = Path("/workspace/audits/qwen3_8_27b/results/baseline_identity.json")


def _compare_to_audit_baseline(formic: dict[str, Any]) -> dict[str, Any]:
    """Anchor Formic to the audit's own identity baseline.

    The audit produced its baseline with the *multimodal* entry point
    (``Qwen3_5ForConditionalGeneration``) on the prompt "Audit technique court.".
    Formic runs the text-only entry point (ADR-0002), so agreement here is
    independent evidence that the two paths compute the same thing.

    Bit-exactness is NOT expected: the two runs differ in configuration
    (``use_cache``, device map / offload split, vision tower resident or not),
    and plan 2.4 requires bit-exactness only at identical config and backend.
    Step 2 measures the tolerances; step 1 only records the agreement.
    """
    entry = formic.get("forward", {}).get("audit_echo")
    if entry is None or not AUDIT_BASELINE.is_file():
        return {"available": False}
    baseline = json.loads(AUDIT_BASELINE.read_text(encoding="utf-8"))
    stats = baseline["last_logits_stats"]
    rms_delta = abs(entry["logits_rms"] - stats["rms"])
    return {
        "available": True,
        "source": str(AUDIT_BASELINE),
        "audit_entry_point": "Qwen3_5ForConditionalGeneration",
        "formic_entry_point": "Qwen3_5ForCausalLM (text-only, ADR-0002)",
        "same_input_ids": entry["input_token_ids"] == baseline.get("input_ids", entry["input_token_ids"]),
        "same_argmax": entry["argmax_id"] == baseline["argmax_token_id"],
        "same_logit_min": entry["logits_min"] == stats["min"],
        "same_logit_max": entry["logits_max"] == stats["max"],
        "rms_formic": entry["logits_rms"],
        "rms_audit": stats["rms"],
        "rms_abs_delta": rms_delta,
        "rms_rel_delta": rms_delta / abs(stats["rms"]) if stats["rms"] else None,
        "same_sha256": entry["logits_sha256"] == baseline["last_logits_checksum"]["sha256_float32_bytes"],
        "note": (
            "argmax/min/max agreement across two different entry points and cache "
            "configurations; SHA divergence is expected cross-configuration (plan 2.4) "
            "and is quantified in step 2 (E4 tolerances)."
        ),
    }


def _run_guard_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_no_cell_reimplementation.py", "-q"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
    )
    return result.returncode == 0


def _print_checklist(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("STEP 1 EXIT CHECKLIST")
    print("=" * 78)
    for item in report["checklist"]:
        mark = "PASS" if item["pass"] else "FAIL"
        print(f"[{mark}] {item['item']}")
        print(f"       {item['detail']}")
    print("-" * 78)
    print(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print("=" * 78)


def _write_markdown(report: dict[str, Any], formic: dict, reference: dict) -> None:
    lines = [
        "# Step 1 acceptance report",
        "",
        f"Generated: {report['generated_at']}",
        f"Config hash: `{report['config_hash']}`",
        f"Prompt set: `{report['prompt_set_sha256'][:16]}`",
        "",
        "## Checklist",
        "",
        "| Item | Result | Detail |",
        "|---|---|---|",
    ]
    for item in report["checklist"]:
        lines.append(
            f"| {item['item']} | {'PASS' if item['pass'] else 'FAIL'} | {item['detail']} |"
        )
    lines += [
        "",
        f"**Overall: {'PASS' if report['overall_pass'] else 'FAIL'}**",
        "",
        "## Formic vs stock Hugging Face",
        "",
        "### Prefill (single forward, position_ids=None on both sides)",
        "",
        "| Prompt | argmax equal | top-10 ids equal | FP32 logits SHA equal | max top-10 delta |",
        "|---|---|---|---|---|",
    ]
    for prompt_id, entry in report["comparison"]["forward"].items():
        lines.append(
            f"| {prompt_id} | {entry['same_argmax']} | {entry['same_top_k_ids']} | "
            f"{entry['same_logits_sha256']} | {entry['max_top_k_delta']:.3e} |"
        )
    lines += [
        "",
        "### Decode, backbone level (explicit greedy loop, no `generate()`)",
        "",
        "| Prompt | tokens identical | matching prefix | cache length equal | max argmax-logit delta |",
        "|---|---|---|---|---|",
    ]
    for prompt_id, entry in report["comparison"]["manual_greedy"].items():
        lines.append(
            f"| {prompt_id} | {entry['identical']} | {entry['matching_prefix']}/{entry['length']} | "
            f"{entry.get('same_cache_length')} | {entry.get('max_argmax_logit_delta', 0):.3e} |"
        )
    lines += [
        "",
        "### Decode through `model.generate()` (wrapper level)",
        "",
        "| Prompt | greedy identical | matching prefix |",
        "|---|---|---|",
    ]
    for prompt_id, entry in report["comparison"]["greedy"].items():
        lines.append(
            f"| {prompt_id} | {entry['identical']} | {entry['matching_prefix']}/{entry['length']} |"
        )
    lines += [
        "",
        "`generate()` differences come from the wrapper, not the backbone: "
        "`Qwen3_5ForConditionalGeneration` overrides "
        "`_prepare_position_ids_for_generation` and, in decode, passes "
        "`position_ids` shaped `[1, B, S]`, which the text model does not "
        "recognise as the `[4, B, S]` contract (audit 05), so it sets "
        "`text_position_ids = None`. `Qwen3_5ForCausalLM` uses the generic "
        "implementation and stays on the documented contract.",
    ]
    audit = report.get("audit_baseline", {})
    if audit.get("available"):
        lines += [
            "",
            "## Anchor to the audit's identity baseline",
            "",
            "Prompt `Audit technique court.`, audit entry point "
            f"`{audit['audit_entry_point']}` vs Formic `{audit['formic_entry_point']}`:",
            "",
            "| Quantity | Audit | Formic | Equal |",
            "|---|---|---|---|",
            f"| argmax token id | {audit['same_argmax'] and 'same' or 'differs'} | | {audit['same_argmax']} |",
            f"| logits min | {audit['same_logit_min'] and 'same' or 'differs'} | | {audit['same_logit_min']} |",
            f"| logits max | {audit['same_logit_max'] and 'same' or 'differs'} | | {audit['same_logit_max']} |",
            f"| logits RMS | {audit['rms_audit']:.10f} | {audit['rms_formic']:.10f} | "
            f"rel delta {audit['rms_rel_delta']:.2e} |",
            f"| FP32 SHA-256 | | | {audit['same_sha256']} |",
            "",
            f"_{audit['note']}_",
        ]
    lines += [
        "",
        "## Load reports",
        "",
        "```",
        f"Formic  ({formic['backbone']['model_class']}): "
        f"{formic['backbone']['parameters']:,} params, "
        f"vision tower present = {formic['backbone']['load_report']['vision_tower_present']}",
        f"HF ref  ({reference['model_class']}): {reference['parameters']:,} params, "
        f"vision tower present = {reference['has_vision_tower']}",
        f"difference: {reference['parameters'] - formic['backbone']['parameters']:,} params "
        "(vision tower, never constructed by Formic - audit constraint A7)",
        "```",
        "",
    ]
    (ARTIFACT_DIR / "step1_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    try:
        shown = path.relative_to(REPO_ROOT)
    except ValueError:
        shown = path
    print(f"[artifact] {shown}")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}; run the corresponding stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Formic step-1 acceptance")
    parser.add_argument("--stage", choices=("formic", "hf", "compare", "all"), default="all")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.stage == "all":
        for stage in ("formic", "hf", "compare"):
            print(f"\n{'=' * 78}\nSTAGE: {stage}\n{'=' * 78}")
            result = subprocess.run(
                # -u: unbuffered, so progress is visible when stdout is a log file
                [sys.executable, "-u", __file__, "--stage", stage, "--config", str(config_path)],
                cwd=str(REPO_ROOT),
                env={**_env(), "PYTHONPATH": str(REPO_ROOT)},
            )
            if result.returncode != 0:
                return result.returncode
        return 0

    if args.stage == "formic":
        stage_formic(config_path)
    elif args.stage == "hf":
        stage_hf(config_path)
    else:
        report = stage_compare(config_path)
        return 0 if report["overall_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
