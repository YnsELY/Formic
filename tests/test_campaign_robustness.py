"""Robustness guards added after the a40-2026-08-26-r1 campaign failure.

These tests are weight-free: no checkpoint, no CUDA, no model forward.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from formic.science.identity.artifacts import (
    ArtifactError,
    CampaignIdentity,
    IncrementalCampaignWriter,
)
from formic.science.identity.campaign import (
    CampaignError,
    _adjudicate_snapshot_candidate,
    _assert_final_gates,
    _assert_stable,
    _phase_continuations,
    _reference_floor_maximum,
    _require_candidate_pass,
)
from formic.science.identity.protocol import InvalidMeasurement
from formic.science.identity.crossover_diagnostic import (
    AttemptMemoryWriter,
    assert_resumable_terminal,
    prepare_attempt_metadata,
)


def _identity() -> CampaignIdentity:
    return CampaignIdentity(
        protocol="SPEC-02-h8-option-b-balanced-v2",
        config_sha256="a" * 64,
        corpus_sha256="b" * 64,
        git_commit="c" * 40,
        backbone_sha256="d" * 64,
    )


def test_require_candidate_pass_raises_on_hard_failure():
    _require_candidate_pass({"verdict": "CANDIDATE_PASS", "reason": "ok"})
    with pytest.raises(CampaignError, match="terminates as FAIL"):
        _require_candidate_pass({"verdict": "FAIL", "reason": "top1_agreement"})
    with pytest.raises(CampaignError):
        _require_candidate_pass({})


def test_final_gates_report_every_failure_together():
    passing = {"verdict": "CANDIDATE_PASS"}
    _assert_final_gates(passing, {"verdict": "CANDIDATE_PASS", "reason": "ok"})

    with pytest.raises(CampaignError) as excinfo:
        _assert_final_gates(
            {"verdict": "FAIL"},
            {"verdict": "FAIL", "reason": "top1_agreement"},
        )
    message = str(excinfo.value)
    assert "snapshot/restore candidate adjudication failed" in message
    assert "top1_agreement" in message
    assert "written before this failure" in message

    with pytest.raises(CampaignError, match="adjudication failed"):
        _assert_final_gates({"verdict": "FAIL"}, {"verdict": "CANDIDATE_PASS"})
    with pytest.raises(CampaignError, match="candidate verdict"):
        _assert_final_gates(passing, {"verdict": "FAIL", "reason": "top1_agreement"})


def test_assert_stable_names_the_changed_side():
    with pytest.raises(InvalidMeasurement, match="changed_side=candidate"):
        _assert_stable([("ref", "cand-a"), ("ref", "cand-b")], "case")
    with pytest.raises(InvalidMeasurement, match="changed_side=reference\\+candidate"):
        _assert_stable([("ref-a", "cand-a"), ("ref-b", "cand-b")], "case")
    with pytest.raises(InvalidMeasurement, match="insufficient_repetitions"):
        _assert_stable([("ref", "cand")], "case")
    _assert_stable([("ref", "cand"), ("ref", "cand")], "case")


def test_campaign_terminal_guard_refuses_completed_run(tmp_path):
    terminal = tmp_path / "terminal.json"

    def guard() -> None:
        assert_resumable_terminal(
            terminal,
            complete_statuses=("CALIBRATION_COMPLETE",),
            resumable_statuses=("FAIL",),
            kind="campaign",
        )

    guard()  # absent terminal is resumable
    terminal.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    guard()
    terminal.write_text(
        json.dumps({"status": "CALIBRATION_COMPLETE"}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="already CALIBRATION_COMPLETE"):
        guard()
    terminal.write_text(json.dumps({"status": "MEASURING"}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="not resumable"):
        guard()


def test_campaign_attempt_metadata_appends_instead_of_overwriting(tmp_path):
    path = tmp_path / "run_metadata.json"
    base = {
        "schema_version": 1,
        "identity": _identity().__dict__,
        "sampled_continuation_seed": 0,
    }
    _, first = prepare_attempt_metadata(
        path, base, {"started_at": "t0", "resume": False, "environment": {}}
    )
    _, second = prepare_attempt_metadata(
        path, base, {"started_at": "t1", "resume": True, "environment": {}}
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert first == "attempt_000" and second == "attempt_001"
    assert [item["attempt_id"] for item in value["attempts"]] == [
        "attempt_000",
        "attempt_001",
    ]
    assert value["identity"] == _identity().__dict__

    changed = dict(base, sampled_continuation_seed=1)
    with pytest.raises(ArtifactError, match="identity or protocol differs"):
        prepare_attempt_metadata(
            path, changed, {"started_at": "t2", "resume": True, "environment": {}}
        )


def test_attempt_memory_writer_appends_live_summaries(tmp_path):
    model = torch.nn.Linear(2, 2)
    writer_a = AttemptMemoryWriter(tmp_path / "cuda_memory.json", "attempt_000")
    writer_a.write_live_summary(model)
    writer_b = AttemptMemoryWriter(tmp_path / "cuda_memory.json", "attempt_001")
    writer_b.write_live_summary(model)

    value = json.loads((tmp_path / "live_tensors.json").read_text(encoding="utf-8"))
    assert set(value["attempts"]) == {"attempt_000", "attempt_001"}


def test_attempt_memory_writer_wraps_legacy_live_summary(tmp_path):
    legacy = {"storage_bytes_by_category": {}, "largest_other_storages": []}
    (tmp_path / "live_tensors.json").write_text(json.dumps(legacy), encoding="utf-8")
    writer = AttemptMemoryWriter(tmp_path / "cuda_memory.json", "attempt_001")
    writer.write_live_summary(torch.nn.Linear(2, 2))

    value = json.loads((tmp_path / "live_tensors.json").read_text(encoding="utf-8"))
    assert value["attempts"]["attempt_000_legacy"] == legacy
    assert "attempt_001" in value["attempts"]


class _ForbiddenSession:
    """A session whose model paths must never run during a pure resume."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            identity=SimpleNamespace(
                decode_prompt_ids=("short_error_assertion",),
                continuation_seeds=(0, 1, 2),
            )
        )

    def greedy_continuation(self, prompt):  # pragma: no cover - must not run
        raise AssertionError("greedy continuation regenerated during resume")

    def sampled_continuation(self, prompt, seed):  # pragma: no cover - must not run
        raise AssertionError("sampled continuation regenerated during resume")


def test_continuations_resume_reuses_committed_cases(tmp_path):
    writer = IncrementalCampaignWriter(tmp_path / "run", _identity())
    prompt = SimpleNamespace(id="short_error_assertion")
    corpus = SimpleNamespace(prompts=(prompt,))
    for case_id, seed, tokens in (
        ("continuation__short_error_assertion__greedy", None, (1, 2, 3, 4, 5, 6, 7, 8)),
        ("continuation__short_error_assertion__s0", 0, (11,) * 8),
        ("continuation__short_error_assertion__s1", 1, (12,) * 8),
        ("continuation__short_error_assertion__s2", 2, (13,) * 8),
    ):
        writer.write_case(
            case_id,
            {
                "schema_version": 1,
                "phase": "reference_continuations",
                "prompt_id": prompt.id,
                "sampling": "greedy" if seed is None else "seeded_sampling",
                "seed": seed,
                "token_ids": list(tokens),
            },
        )

    result = _phase_continuations(writer, _ForbiddenSession(), corpus)

    assert result[("short_error_assertion", None)] == (1, 2, 3, 4, 5, 6, 7, 8)
    assert result[("short_error_assertion", 2)] == (13,) * 8
    assert "reference_continuations" in writer.completed_phases()


def _snapshot(delta: float) -> dict:
    return {
        "observations": [
            {
                "repetition": 0,
                "comparisons": [
                    {"tensor": {"max_abs_delta": delta}, "top1_agreement": True}
                ],
            }
        ],
        "stability": {"last_two_exact": True},
    }


def _candidate(threshold: float) -> dict:
    return {
        "records": [
            {
                "mode": "decode_cached",
                "point": "logits",
                "length_class": "short",
                "max_abs_delta": threshold,
            }
        ]
    }


def test_snapshot_adjudication_never_demands_less_than_the_reference_floor():
    # An exact calibration row (0.0) must not fail a restore whose delta sits
    # under the measured backend repeat noise.
    result = _adjudicate_snapshot_candidate(
        _snapshot(0.25), _candidate(0.0), reference_floor_max_abs_delta=0.3
    )
    assert result["verdict"] == "CANDIDATE_PASS"
    assert result["max_abs_delta"] == 0.3
    assert result["threshold_source"] == "reference_floor"
    assert result["candidate_row_max_abs_delta"] == 0.0

    # The floor never loosens a candidate row that is already wider.
    wide = _adjudicate_snapshot_candidate(
        _snapshot(0.25), _candidate(0.5), reference_floor_max_abs_delta=0.3
    )
    assert wide["max_abs_delta"] == 0.5
    assert wide["threshold_source"] == "candidate_row"

    # Without a floor the behaviour is unchanged.
    strict = _adjudicate_snapshot_candidate(_snapshot(0.25), _candidate(0.0))
    assert strict["verdict"] == "FAIL"
    assert strict["reference_floor_max_abs_delta"] is None


def test_reference_floor_maximum_extracts_the_rr_maximum():
    cases = [
        {
            "prompt_id": "audit_echo",
            "raw_control_floor": [
                {
                    "pair": "reference_reference",
                    "repetition": 0,
                    "step": 1,
                    "metric": {"max_abs_delta": 0.125},
                },
                {
                    "pair": "runner_runner",
                    "repetition": 0,
                    "step": 1,
                    "metric": {"max_abs_delta": 9.0},
                },
                {
                    "pair": "reference_reference",
                    "repetition": 1,
                    "step": 2,
                    "metric": {"max_abs_delta": 0.5},
                },
            ],
        }
    ]
    assert _reference_floor_maximum(cases) == 0.5
    assert _reference_floor_maximum([{"prompt_id": "x", "raw_control_floor": []}]) is None
