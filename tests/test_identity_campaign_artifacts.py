from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from formic.science.identity.artifacts import ArtifactError, CampaignIdentity, IncrementalCampaignWriter
from formic.science.identity.campaign import _MeasurementSession, _measure_or_resume, _stability_details
from formic.science.identity.protocol import InvalidMeasurement
from formic.science.identity.types import SamplingMode


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


def test_campaign_writer_persists_mutable_diagnostic_without_completing_case(tmp_path):
    writer = IncrementalCampaignWriter(tmp_path / "run", _identity())
    writer.write_diagnostic("legacy__audit_echo", {"status": "MEASURING", "repetitions": [0]})
    writer.write_diagnostic("legacy__audit_echo", {"status": "FAILED", "repetitions": [0, 1]})

    payload = json.loads((writer.diagnostics_dir / "legacy__audit_echo.json").read_text())
    assert payload == {"status": "FAILED", "repetitions": [0, 1]}
    assert writer.completed_cases() == frozenset()


def test_measurement_failure_persists_last_repetition_diagnostic(tmp_path):
    class FailingSession:
        def measure_forced(self, **kwargs):
            kwargs["repetition_observer"](
                {
                    "schema_version": 1,
                    "status": "MEASURING",
                    "case_id": "legacy__audit_echo",
                    "stability": {"last_two_exact": False},
                }
            )
            raise InvalidMeasurement("last two measured traces are unstable")

    writer = IncrementalCampaignWriter(tmp_path / "run", _identity())
    with pytest.raises(InvalidMeasurement, match="unstable"):
        _measure_or_resume(
            writer,
            "legacy__audit_echo",
            "legacy_continuity",
            FailingSession(),
            object(),
            forced_token_ids=(),
            repetitions=3,
            sampling=SamplingMode.GREEDY,
            continuation_seed=None,
            exact_required=True,
        )

    payload = json.loads((writer.diagnostics_dir / "legacy__audit_echo.json").read_text())
    assert payload["status"] == "FAILED"
    assert payload["failure"]["exception"] == "InvalidMeasurement"
    assert payload["stability"]["last_two_exact"] is False


def test_stability_details_identify_first_changed_repetition():
    details = _stability_details([("ref-a", "run-a"), ("ref-a", "run-a"), ("ref-b", "run-b")])

    assert details["last_two_exact"] is False
    assert details["first_changed_repetition"] == 2
    assert details["fingerprints"][2] == {
        "repetition": 2,
        "reference_fingerprint": "ref-b",
        "candidate_fingerprint": "run-b",
    }


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
