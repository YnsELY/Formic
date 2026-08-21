from __future__ import annotations

import json

import pytest

from formic.science.identity.artifacts import (
    ArtifactError,
    IncrementalRunWriter,
    RunIdentity,
)


def _identity(commit="abc"):
    return RunIdentity("run-1", "c" * 64, "t" * 64, commit, "b" * 64)


def test_prompt_checkpoints_are_incremental_and_resumable(tmp_path):
    writer = IncrementalRunWriter(tmp_path / "run", _identity())
    writer.write_prompt("short_a", {"verdict": "PASS", "measurements": [1]})
    assert writer.completed_prompt_ids() == {"short_a"}

    resumed = IncrementalRunWriter(tmp_path / "run", _identity())
    assert resumed.completed_prompt_ids() == {"short_a"}
    resumed.write_prompt("medium_a", {"verdict": "PASS", "measurements": [2]})
    resumed.validate()
    manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["completed"]) == {"short_a", "medium_a"}


def test_resume_refuses_identity_or_completed_payload_changes(tmp_path):
    writer = IncrementalRunWriter(tmp_path / "run", _identity())
    writer.write_prompt("p", {"value": 1})
    with pytest.raises(ArtifactError, match="identity differs"):
        IncrementalRunWriter(tmp_path / "run", _identity(commit="different"))
    with pytest.raises(ArtifactError, match="changed during resume"):
        writer.write_prompt("p", {"value": 2})


def test_validation_detects_corrupt_prompt_checkpoint(tmp_path):
    writer = IncrementalRunWriter(tmp_path / "run", _identity())
    path = writer.write_prompt("p", {"value": 1})
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="invalid"):
        writer.validate()
