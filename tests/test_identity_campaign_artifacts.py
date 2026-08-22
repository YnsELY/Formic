from __future__ import annotations

import pytest

from formic.science.identity.artifacts import ArtifactError, CampaignIdentity, IncrementalCampaignWriter


def _identity() -> CampaignIdentity:
    return CampaignIdentity(
        protocol="SPEC-02-h8-option-b",
        config_sha256="a" * 64,
        corpus_sha256="b" * 64,
        git_commit="c" * 40,
        backbone_sha256="d" * 64,
    )


def test_campaign_writer_commits_cases_and_phases_atomically_and_resumes(tmp_path):
    writer = IncrementalCampaignWriter(tmp_path / "run", _identity())
    writer.write_case("case_a", {"result": 1})
    writer.write_phase("preflight", {"result": "done"})
    writer.validate()

    resumed = IncrementalCampaignWriter(tmp_path / "run", _identity())
    assert resumed.completed_cases() == {"case_a"}
    assert resumed.completed_phases() == {"preflight"}
    assert resumed.write_case("case_a", {"result": 1}).is_file()
    with pytest.raises(ArtifactError, match="changed"):
        resumed.write_case("case_a", {"result": 2})


def test_campaign_writer_refuses_resume_against_different_sources(tmp_path):
    IncrementalCampaignWriter(tmp_path / "run", _identity())
    changed = CampaignIdentity(**{**_identity().__dict__, "backbone_sha256": "e" * 64})
    with pytest.raises(ArtifactError, match="resume identity"):
        IncrementalCampaignWriter(tmp_path / "run", changed)
