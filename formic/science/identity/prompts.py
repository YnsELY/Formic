"""Strict loader for the immutable 12-prompt SPEC-02 corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

LengthClass = Literal["legacy", "short", "medium", "long"]
LEGACY_IDS = (
    "audit_echo",
    "plain_text",
    "code_completion",
    "code_bugfix",
    "instruction_short",
    "instruction_scope",
)
CLASS_RANGES = {"short": (8, 32), "medium": (256, 512), "long": (2_000, 4_000)}


class PromptCorpusError(ValueError):
    pass


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenPrompt:
    id: str
    set_name: Literal["legacy", "calibration"]
    length_class: LengthClass
    text: str
    token_ids: tuple[int, ...]
    text_sha256: str
    token_ids_sha256: str

    def validate(self) -> None:
        if not self.id or not self.text or not self.token_ids:
            raise PromptCorpusError("prompt id, exact text and token IDs are required")
        if self.set_name == "legacy" and self.length_class != "legacy":
            raise PromptCorpusError(f"legacy prompt {self.id} must stay outside length bands")
        if self.set_name == "calibration" and self.length_class not in CLASS_RANGES:
            raise PromptCorpusError(f"invalid calibration length class for {self.id}")
        if self.text_sha256 != text_sha256(self.text):
            raise PromptCorpusError(f"text hash mismatch for {self.id}")
        if self.token_ids_sha256 != token_ids_sha256(self.token_ids):
            raise PromptCorpusError(f"token ID hash mismatch for {self.id}")
        if self.set_name == "calibration":
            minimum, maximum = CLASS_RANGES[self.length_class]
            if not minimum <= len(self.token_ids) <= maximum:
                raise PromptCorpusError(
                    f"{self.id} has {len(self.token_ids)} tokens, outside "
                    f"{self.length_class} range {minimum}-{maximum}"
                )


@dataclass(frozen=True)
class FrozenPromptCorpus:
    schema_version: int
    status: Literal["frozen"]
    tokenizer_repo_id: str
    tokenizer_revision: str
    approved_by: str
    approved_on: str
    change_policy: Literal["ADR_REQUIRED"]
    prompts: tuple[FrozenPrompt, ...]
    corpus_sha256: str
    source_sha256: str

    def validate(self) -> None:
        if self.schema_version != 2 or self.status != "frozen":
            raise PromptCorpusError("prompt corpus must be frozen schema version 2")
        if not self.approved_by or not self.approved_on or self.change_policy != "ADR_REQUIRED":
            raise PromptCorpusError("frozen corpus requires human review and ADR change policy")
        if len(self.prompts) != 12 or len({item.id for item in self.prompts}) != 12:
            raise PromptCorpusError("SPEC-02 corpus must contain 12 unique prompts")
        for item in self.prompts:
            item.validate()
        legacy = tuple(item.id for item in self.prompts if item.set_name == "legacy")
        if legacy != LEGACY_IDS:
            raise PromptCorpusError("legacy prompt IDs/order changed")
        calibration = [item for item in self.prompts if item.set_name == "calibration"]
        counts = {name: 0 for name in CLASS_RANGES}
        for item in calibration:
            counts[item.length_class] += 1
        if counts != {"short": 2, "medium": 2, "long": 2}:
            raise PromptCorpusError(f"calibration class cardinality changed: {counts}")
        if self.corpus_sha256 != corpus_sha256(self.prompts):
            raise PromptCorpusError("corpus hash mismatch")

    def validate_tokenizer(self, tokenizer: Any) -> None:
        for prompt in self.prompts:
            actual = tuple(tokenizer(prompt.text, add_special_tokens=False)["input_ids"])
            if actual != prompt.token_ids:
                raise PromptCorpusError(f"tokenizer output changed for {prompt.id}")


def corpus_sha256(prompts: tuple[FrozenPrompt, ...]) -> str:
    payload = [
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
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_corpus(path: str | Path) -> FrozenPromptCorpus:
    source = Path(path)
    raw_bytes = source.read_bytes()
    value = yaml.safe_load(raw_bytes)
    _strict(
        value,
        {"schema_version", "status", "review", "tokenizer", "prompts", "corpus_sha256"},
        "corpus",
    )
    review = value["review"]
    _strict(review, {"approved_by", "approved_on", "change_policy"}, "corpus.review")
    tokenizer = value["tokenizer"]
    _strict(tokenizer, {"repo_id", "revision"}, "corpus.tokenizer")
    prompts = tuple(_prompt(item, index) for index, item in enumerate(value["prompts"]))
    result = FrozenPromptCorpus(
        schema_version=value["schema_version"],
        status=value["status"],
        tokenizer_repo_id=tokenizer["repo_id"],
        tokenizer_revision=tokenizer["revision"],
        approved_by=review["approved_by"],
        approved_on=review["approved_on"],
        change_policy=review["change_policy"],
        prompts=prompts,
        corpus_sha256=value["corpus_sha256"],
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    result.validate()
    return result


def _prompt(value: dict[str, Any], index: int) -> FrozenPrompt:
    path = f"prompts[{index}]"
    _strict(
        value,
        {"id", "set", "length_class", "text", "token_ids", "text_sha256", "token_ids_sha256"},
        path,
    )
    return FrozenPrompt(
        id=value["id"],
        set_name=value["set"],
        length_class=value["length_class"],
        text=value["text"],
        token_ids=tuple(value["token_ids"]),
        text_sha256=value["text_sha256"],
        token_ids_sha256=value["token_ids_sha256"],
    )


def _strict(value: Any, allowed: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise PromptCorpusError(f"{path} must be a mapping")
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise PromptCorpusError(f"{path}: missing={missing}, unknown={unknown}")
