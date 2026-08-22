from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from formic.science.identity.artifacts import ArtifactError, CampaignIdentity, IncrementalCampaignWriter
from formic.science.identity.campaign import _MeasurementSession


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


def test_trace_inertness_warmup_disables_autograd():
    class GradRecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.grad_enabled = []

        def forward(self, *, input_ids, use_cache):
            self.grad_enabled.append(torch.is_grad_enabled())
            return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[-1], 4))

    model = GradRecordingModel()
    handle = SimpleNamespace(model=model, view=None, device=torch.device("cpu"))
    config = SimpleNamespace(numerics=SimpleNamespace(warmup_traces_per_shape=1))
    session = _MeasurementSession(handle, config)

    session.trace_off_prefill(SimpleNamespace(token_ids=(1, 2, 3)))

    assert model.grad_enabled == [False]
