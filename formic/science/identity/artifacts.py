"""Incremental, resumable identity-run artefacts.

One atomic checkpoint is committed to disk after each prompt. Warmups never
enter this writer, so an interrupted run retains every completed measurement
without retaining discarded state captures in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return sha256_bytes(payload)


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    config_sha256: str
    tolerances_sha256: str
    git_commit: str
    backbone_sha256: str


class IncrementalRunWriter:
    """Write immutable prompt checkpoints plus an atomically updated manifest."""

    def __init__(self, root: str | Path, identity: RunIdentity):
        self.root = Path(root)
        self.identity = identity
        self.prompts_dir = self.root / "prompts"
        self.manifest_path = self.root / "manifest.json"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = self._read_manifest()
            if manifest["identity"] != identity.__dict__:
                raise ArtifactError("resume identity differs from existing run manifest")
        else:
            atomic_write_json(
                self.manifest_path,
                {"schema_version": 1, "identity": identity.__dict__, "completed": {}},
            )

    def completed_prompt_ids(self) -> frozenset[str]:
        return frozenset(self._read_manifest()["completed"])

    def write_prompt(self, prompt_id: str, payload: dict[str, Any]) -> Path:
        if not prompt_id or any(char in prompt_id for char in "/\\"):
            raise ArtifactError("prompt_id must be a non-empty filename-safe identifier")
        manifest = self._read_manifest()
        target = self.prompts_dir / f"{prompt_id}.json"
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        existing = manifest["completed"].get(prompt_id)
        if existing is not None and existing != digest:
            raise ArtifactError(f"completed prompt {prompt_id!r} changed during resume")
        if existing is not None:
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ArtifactError(f"completed prompt checkpoint is missing or corrupt: {prompt_id}")
            return target
        atomic_write_json(target, payload)
        manifest["completed"][prompt_id] = digest
        atomic_write_json(self.manifest_path, manifest)
        return target

    def validate(self) -> None:
        manifest = self._read_manifest()
        for prompt_id, expected in manifest["completed"].items():
            path = self.prompts_dir / f"{prompt_id}.json"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ArtifactError(f"prompt checkpoint invalid: {prompt_id}")

    def _read_manifest(self) -> dict[str, Any]:
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if set(value) != {"schema_version", "identity", "completed"}:
            raise ArtifactError("invalid run manifest schema")
        if value["schema_version"] != 1 or not isinstance(value["completed"], dict):
            raise ArtifactError("invalid run manifest")
        return value
