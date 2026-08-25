from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from formic.science.identity.budget import (
    EXPECTED_PHASE_FORWARDS,
    PREFLIGHT_FORWARDS,
    PROTOCOL_ID,
    EstimateError,
    load_preflight_estimate,
    report_estimate,
)


def _value(seconds: float = 100.0):
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "model_processes": 1,
        "model_load_seconds": seconds,
        "preflight_forwards": PREFLIGHT_FORWARDS,
        "preflight_elapsed_seconds": seconds,
        "phases": [
            {"name": name, "forwards": forwards, "estimated_seconds": seconds}
            for name, forwards in EXPECTED_PHASE_FORWARDS.items()
        ],
    }


def _load(tmp_path, value):
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return load_preflight_estimate(path)


def test_estimate_reports_one_load_and_all_phase_durations(tmp_path):
    report = report_estimate(_load(tmp_path, _value()))
    assert report.model_processes == 1
    assert report.model_load_seconds == 100.0
    assert report.remaining_estimated_seconds == 900.0
    assert report.total_estimated_seconds == 1_000.0
    assert report.total_forwards == 8_549
    assert [phase.name for phase in report.phases] == list(EXPECTED_PHASE_FORWARDS)


def test_estimate_rejects_changed_model_process_count_or_forward_plan(tmp_path):
    value = _value()
    value["model_processes"] = 2
    with pytest.raises(EstimateError, match="one model process"):
        _load(tmp_path, value)

    value = _value()
    value["phases"][0]["forwards"] += 1
    with pytest.raises(EstimateError, match="forward plan changed"):
        _load(tmp_path, value)


def test_estimator_script_never_blocks_the_session(tmp_path):
    source = tmp_path / "preflight.json"
    output = tmp_path / "estimate.json"
    source.write_text(json.dumps(_value()), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "step2_budget_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--preflight", str(source), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "TOTAL ESTIMATED" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["report"] == "ESTIMATE"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preflight",
            str(source),
            "--output",
            str(output),
            "--budget-hours",
            "0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "ignored unsupported arguments" in completed.stdout

    source.write_text("not json", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), "--preflight", str(source), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "ERROR" in completed.stdout
