#!/usr/bin/env python3
"""Materialise the human-approved immutable SPEC-02 prompt corpus."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from formic.science.identity.prompts import (  # noqa: E402
    FrozenPrompt,
    corpus_sha256,
    text_sha256,
    token_ids_sha256,
)

TOKENIZER_PATH = REPO_ROOT / "artifacts" / "step2" / "hf_nonweights"
LEGACY_PATH = REPO_ROOT / "configs" / "reference_prompts_legacy_v1.yaml"
CANDIDATES_PATH = REPO_ROOT / "configs" / "reference_prompt_candidates.yaml"
OUTPUT_PATH = REPO_ROOT / "configs" / "reference_prompts.yaml"


class LiteralDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


LiteralDumper.add_representer(str, _str_representer)


def _render_legacy(tokenizer) -> list[tuple[str, str]]:
    source = yaml.safe_load(LEGACY_PATH.read_text(encoding="utf-8"))
    result = []
    for item in source["prompts"]:
        if item["kind"] == "raw":
            text = item["text"]
        else:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        result.append((item["id"], text))
    return result


def main() -> int:
    from transformers import AutoTokenizer

    candidates = yaml.safe_load(CANDIDATES_PATH.read_text(encoding="utf-8"))
    if candidates.get("status") != "REVIEWED_SOURCE_FROZEN_AS_SCHEMA_V2":
        raise RuntimeError("unexpected candidate source status")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    prompts: list[FrozenPrompt] = []
    for prompt_id, text in _render_legacy(tokenizer):
        token_ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        prompts.append(
            FrozenPrompt(
                prompt_id,
                "legacy",
                "legacy",
                text,
                token_ids,
                text_sha256(text),
                token_ids_sha256(token_ids),
            )
        )
    for item in candidates["prompts"]:
        text = item["text"]
        if text_sha256(text) != item["text_sha256"]:
            raise RuntimeError(f"reviewed candidate text changed: {item['id']}")
        token_ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) != item["exact_token_count"]:
            raise RuntimeError(f"reviewed candidate token count changed: {item['id']}")
        prompts.append(
            FrozenPrompt(
                item["id"],
                "calibration",
                item["length_class"],
                text,
                token_ids,
                text_sha256(text),
                token_ids_sha256(token_ids),
            )
        )
    frozen = tuple(prompts)
    records = [
        {
            "id": item.id,
            "set": item.set_name,
            "length_class": item.length_class,
            "text": item.text,
            "token_ids": list(item.token_ids),
            "text_sha256": item.text_sha256,
            "token_ids_sha256": item.token_ids_sha256,
        }
        for item in frozen
    ]
    payload = {
        "schema_version": 2,
        "status": "frozen",
        "review": {
            "approved_by": "Yanis",
            "approved_on": "2026-08-21",
            "change_policy": "ADR_REQUIRED",
        },
        "tokenizer": {
            "repo_id": candidates["tokenizer"]["repo_id"],
            "revision": candidates["tokenizer"]["revision"],
        },
        "prompts": records,
        "corpus_sha256": corpus_sha256(frozen),
    }
    header = (
        "# FROZEN SPEC-02 corpus. Any change requires an ADR and invalidates prior verdicts.\n"
        f"# Generated from reviewed source SHA-256: {hashlib.sha256(CANDIDATES_PATH.read_bytes()).hexdigest()}\n"
    )
    OUTPUT_PATH.write_text(
        header
        + yaml.dump(payload, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"corpus_sha256={payload['corpus_sha256']}")
    for item in frozen:
        print(f"{item.id}: {len(item.token_ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
