from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from formic.science.identity.prompts import (
    LEGACY_IDS,
    FrozenPrompt,
    PromptCorpusError,
    corpus_sha256,
    load_frozen_corpus,
    text_sha256,
    token_ids_sha256,
)


def _prompt(prompt_id, set_name, length_class, length):
    text = f"exact text for {prompt_id}"
    ids = tuple(range(length))
    return FrozenPrompt(
        prompt_id,
        set_name,
        length_class,
        text,
        ids,
        text_sha256(text),
        token_ids_sha256(ids),
    )


def _value():
    prompts = tuple(
        [_prompt(prompt_id, "legacy", "legacy", 4) for prompt_id in LEGACY_IDS]
        + [_prompt(f"short_{index}", "calibration", "short", 8) for index in (1, 2)]
        + [_prompt(f"medium_{index}", "calibration", "medium", 256) for index in (1, 2)]
        + [_prompt(f"long_{index}", "calibration", "long", 2000) for index in (1, 2)]
    )
    return {
        "schema_version": 2,
        "status": "frozen",
        "review": {
            "approved_by": "Yanis",
            "approved_on": "2026-08-21",
            "change_policy": "ADR_REQUIRED",
        },
        "tokenizer": {"repo_id": "Qwen/Qwen3.8-27B", "revision": "abc"},
        "prompts": [
            {
                "id": item.id,
                "set": item.set_name,
                "length_class": item.length_class,
                "text": item.text,
                "token_ids": list(item.token_ids),
                "text_sha256": item.text_sha256,
                "token_ids_sha256": item.token_ids_sha256,
            }
            for item in prompts
        ],
        "corpus_sha256": corpus_sha256(prompts),
    }


def _write(tmp_path, value):
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_frozen_prompt_corpus_validates_cardinality_ranges_and_hashes(tmp_path):
    corpus = load_frozen_corpus(_write(tmp_path, _value()))
    assert len(corpus.prompts) == 12
    assert len(corpus.source_sha256) == 64


def test_text_or_token_id_mutation_is_fatal(tmp_path):
    value = _value()
    value["prompts"][0]["text"] += " changed"
    with pytest.raises(PromptCorpusError, match="text hash"):
        load_frozen_corpus(_write(tmp_path, value))

    value = _value()
    value["prompts"][6]["token_ids"] = [1, 2]
    with pytest.raises(PromptCorpusError, match="token ID hash"):
        load_frozen_corpus(_write(tmp_path, value))


def test_wrong_class_cardinality_or_unknown_fields_are_fatal(tmp_path):
    value = _value()
    value["prompts"][6]["length_class"] = "medium"
    with pytest.raises(PromptCorpusError):
        load_frozen_corpus(_write(tmp_path, value))
    value = _value()
    value["unreviewed"] = True
    with pytest.raises(PromptCorpusError, match="unknown"):
        load_frozen_corpus(_write(tmp_path, value))


def test_committed_spec02_corpus_is_the_human_approved_immutable_version():
    path = Path(__file__).resolve().parents[1] / "configs" / "reference_prompts.yaml"
    corpus = load_frozen_corpus(path)
    assert corpus.approved_by == "Yanis"
    assert corpus.change_policy == "ADR_REQUIRED"
    assert corpus.corpus_sha256 == (
        "482e63d88a53d2850fe87db648f7d6fe2414ca5ee64b1a307de7cb3501c1f3c0"
    )
    assert [(item.id, len(item.token_ids)) for item in corpus.prompts] == [
        ("audit_echo", 4),
        ("plain_text", 5),
        ("code_completion", 20),
        ("code_bugfix", 24),
        ("instruction_short", 86),
        ("instruction_scope", 86),
        ("short_error_assertion", 26),
        ("short_diff_review", 25),
        ("medium_cache_regression", 310),
        ("medium_scoped_diff", 331),
        ("long_resume_incidents", 2437),
        ("long_monorepo_diff", 2542),
    ]
