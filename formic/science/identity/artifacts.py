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


@dataclass(frozen=True)
class CampaignIdentity:
    """Sources pinned before the first measured GPU phase.

    ``tolerances.json`` deliberately does not appear here: the first A40 run
    is a calibration and must not pretend that a threshold table exists before
    its raw observations have been written and reviewed.
    """

    protocol: str
    config_sha256: str
    corpus_sha256: str
    git_commit: str
    backbone_sha256: str


class IncrementalCampaignWriter:
    """Atomic, resumable writer for the complete GPU campaign.

    A phase is committed only after all of its case files have been committed.
    Resume therefore never treats a partially written phase as complete.
    """

    def __init__(self, root: str | Path, identity: CampaignIdentity):
        self.root = Path(root)
        self.identity = identity
        self.cases_dir = self.root / "prompts"
        self.phases_dir = self.root / "phases"
        # Diagnostics are deliberately outside the resumable manifest: they
        # are updated after each measured repetition, including for a case
        # that is invalid and must therefore never be marked completed.
        self.diagnostics_dir = self.root / "diagnostics"
        self.manifest_path = self.root / "manifest.json"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.phases_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = self._read_manifest()
            if manifest["identity"] != identity.__dict__:
                raise ArtifactError("resume identity differs from existing campaign")
        else:
            atomic_write_json(
                self.manifest_path,
                {
                    "schema_version": 1,
                    "identity": identity.__dict__,
                    "completed_cases": {},
                    "completed_phases": {},
                },
            )

    def completed_cases(self) -> frozenset[str]:
        return frozenset(self._read_manifest()["completed_cases"])

    def completed_phases(self) -> frozenset[str]:
        return frozenset(self._read_manifest()["completed_phases"])

    def write_case(self, case_id: str, payload: dict[str, Any]) -> Path:
        return self._write(
            directory=self.cases_dir,
            manifest_key="completed_cases",
            item_id=case_id,
            payload=payload,
        )

    def write_phase(self, phase: str, payload: dict[str, Any]) -> Path:
        return self._write(
            directory=self.phases_dir,
            manifest_key="completed_phases",
            item_id=phase,
            payload=payload,
        )

    def write_diagnostic(self, case_id: str, payload: dict[str, Any]) -> Path:
        """Atomically replace the latest partial measurement for ``case_id``.

        Unlike a completed case this record may change as repetitions arrive.
        It is intentionally not resumable evidence and never advances the
        manifest, but it makes a stability failure inspectable after the GPU
        process exits.
        """
        if not case_id or any(char in case_id for char in "/\\"):
            raise ArtifactError("campaign item id must be filename-safe")
        target = self.diagnostics_dir / f"{case_id}.json"
        atomic_write_json(target, payload)
        return target

    def validate(self) -> None:
        manifest = self._read_manifest()
        for key, directory in (
            ("completed_cases", self.cases_dir),
            ("completed_phases", self.phases_dir),
        ):
            for item_id, expected in manifest[key].items():
                path = directory / f"{item_id}.json"
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ArtifactError(f"campaign checkpoint invalid: {key}/{item_id}")

    def _write(
        self,
        *,
        directory: Path,
        manifest_key: str,
        item_id: str,
        payload: dict[str, Any],
    ) -> Path:
        if not item_id or any(char in item_id for char in "/\\"):
            raise ArtifactError("campaign item id must be filename-safe")
        manifest = self._read_manifest()
        target = directory / f"{item_id}.json"
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        existing = manifest[manifest_key].get(item_id)
        if existing is not None and existing != digest:
            raise ArtifactError(f"completed campaign item changed during resume: {item_id}")
        if existing is not None:
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ArtifactError(f"campaign checkpoint missing or corrupt: {item_id}")
            return target
        atomic_write_json(target, payload)
        manifest[manifest_key][item_id] = digest
        atomic_write_json(self.manifest_path, manifest)
        return target

    def _read_manifest(self) -> dict[str, Any]:
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = {"schema_version", "identity", "completed_cases", "completed_phases"}
        if set(value) != expected or value["schema_version"] != 1:
            raise ArtifactError("invalid campaign manifest schema")
        if not isinstance(value["completed_cases"], dict) or not isinstance(value["completed_phases"], dict):
            raise ArtifactError("invalid campaign manifest items")
        return value
