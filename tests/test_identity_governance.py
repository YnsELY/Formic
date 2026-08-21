from __future__ import annotations

import hashlib
import json

import pytest

from formic.science.backbone_hash import ALGORITHM
from formic.science.identity.governance import (
    GovernanceError,
    load_gpu_verdict,
    verify_latest_pass,
    verify_tolerance_governance,
)


def _verdict():
    return {
        "schema_version": 1,
        "spec": "SPEC-02",
        "verdict": "PASS",
        "generated_at": "2026-08-21T12:00:00Z",
        "config_sha256": "1" * 64,
        "tolerances_sha256": "2" * 64,
        "corpus_sha256": "3" * 64,
        "git_commit": "abcdef",
        "backbone_algorithm": ALGORITHM,
        "backbone_sha256": "4" * 64,
        "backbone_tensor_count": 851,
        "hf_repo_id": "Qwen/Qwen3.8-27B",
        "hf_revision": "revision",
        "hardware": {"gpu_name": "NVIDIA A40", "gpu_vram_gib": 48},
        "environment": {"torch": "2.4.1"},
        "campaign_artifact": "artifacts/step2/run/manifest.json",
        "campaign_artifact_sha256": "5" * 64,
        "first_divergence": None,
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_pass_verdict_is_fresh_only_for_identical_sources(tmp_path):
    verdict = load_gpu_verdict(_write_json(tmp_path / "verdict.json", _verdict()))
    verify_latest_pass(
        verdict,
        config_sha256="1" * 64,
        tolerances_sha256="2" * 64,
        corpus_sha256="3" * 64,
        backbone_sha256="4" * 64,
    )
    with pytest.raises(GovernanceError, match="config"):
        verify_latest_pass(
            verdict,
            config_sha256="9" * 64,
            tolerances_sha256="2" * 64,
            corpus_sha256="3" * 64,
            backbone_sha256="4" * 64,
        )


def test_hardware_and_fail_diagnostic_are_strict(tmp_path):
    value = _verdict()
    value["hardware"]["gpu_name"] = "NVIDIA H100"
    with pytest.raises(GovernanceError, match="A40"):
        load_gpu_verdict(_write_json(tmp_path / "verdict.json", value))
    value = _verdict()
    value["verdict"] = "FAIL"
    with pytest.raises(GovernanceError, match="first divergence"):
        load_gpu_verdict(_write_json(tmp_path / "verdict.json", value))


def test_current_tolerance_hash_requires_hashed_report_and_adr(tmp_path):
    tolerances = tmp_path / "tolerances.json"
    tolerances.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text("measured\n", encoding="utf-8")
    adr = tmp_path / "adr.md"
    adr.write_text("PROPOSED\n", encoding="utf-8")
    governance = {
        "schema_version": 1,
        "records": [
            {
                "tolerances_sha256": hashlib.sha256(tolerances.read_bytes()).hexdigest(),
                "report": "report.md",
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "adr": "adr.md",
            }
        ],
    }
    path = _write_json(tmp_path / "governance.json", governance)
    verify_tolerance_governance(
        path, tolerances_path=tolerances, repo_root=tmp_path
    )
    report.write_text("changed\n", encoding="utf-8")
    with pytest.raises(GovernanceError, match="report hash"):
        verify_tolerance_governance(
            path, tolerances_path=tolerances, repo_root=tmp_path
        )
