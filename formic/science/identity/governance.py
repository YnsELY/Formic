"""Provenance and freshness checks for the blocking GPU identity verdict."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from formic.science.backbone_hash import ALGORITHM


class GovernanceError(RuntimeError):
    pass


def _is_sha256(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class GpuVerdictArtifact:
    schema_version: int
    spec: str
    verdict: Literal["PASS", "FAIL"]
    generated_at: str
    config_sha256: str
    tolerances_sha256: str
    corpus_sha256: str
    git_commit: str
    backbone_algorithm: str
    backbone_sha256: str
    backbone_tensor_count: int
    hf_repo_id: str
    hf_revision: str
    gpu_name: str
    gpu_vram_gib: int
    environment: dict[str, Any]
    campaign_artifact: str
    campaign_artifact_sha256: str
    first_divergence: dict[str, Any] | None

    def validate(self) -> None:
        if self.schema_version != 1 or self.spec != "SPEC-02":
            raise GovernanceError("invalid GPU verdict schema/spec")
        if self.verdict not in ("PASS", "FAIL"):
            raise GovernanceError("invalid verdict")
        try:
            datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise GovernanceError("invalid verdict date") from exc
        for label, value in (
            ("config", self.config_sha256),
            ("tolerances", self.tolerances_sha256),
            ("corpus", self.corpus_sha256),
            ("backbone", self.backbone_sha256),
            ("campaign", self.campaign_artifact_sha256),
        ):
            if not _is_sha256(value):
                raise GovernanceError(f"invalid {label} SHA-256")
        if self.backbone_algorithm != ALGORITHM or self.backbone_tensor_count != 851:
            raise GovernanceError("verdict does not identify the canonical 851-tensor backbone")
        if self.hf_repo_id != "Qwen/Qwen3.8-27B" or not self.hf_revision:
            raise GovernanceError("verdict checkpoint metadata changed")
        if self.gpu_name != "NVIDIA A40" or self.gpu_vram_gib != 48:
            raise GovernanceError("SPEC-02 final gate requires the pinned NVIDIA A40 48 GB")
        if self.verdict == "PASS" and self.first_divergence is not None:
            raise GovernanceError("PASS verdict cannot contain a first divergence")
        if self.verdict == "FAIL" and self.first_divergence is None:
            raise GovernanceError("FAIL verdict must contain the first divergence")


def load_gpu_verdict(path: str | Path) -> GpuVerdictArtifact:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version", "spec", "verdict", "generated_at", "config_sha256",
        "tolerances_sha256", "corpus_sha256", "git_commit", "backbone_algorithm",
        "backbone_sha256", "backbone_tensor_count", "hf_repo_id", "hf_revision",
        "hardware", "environment", "campaign_artifact", "campaign_artifact_sha256",
        "first_divergence",
    }
    _strict(value, expected, "verdict")
    hardware = value["hardware"]
    _strict(hardware, {"gpu_name", "gpu_vram_gib"}, "verdict.hardware")
    result = GpuVerdictArtifact(
        value["schema_version"], value["spec"], value["verdict"], value["generated_at"],
        value["config_sha256"], value["tolerances_sha256"], value["corpus_sha256"],
        value["git_commit"], value["backbone_algorithm"], value["backbone_sha256"],
        value["backbone_tensor_count"], value["hf_repo_id"], value["hf_revision"],
        hardware["gpu_name"], hardware["gpu_vram_gib"], value["environment"],
        value["campaign_artifact"], value["campaign_artifact_sha256"],
        value["first_divergence"],
    )
    result.validate()
    return result


def verify_latest_pass(
    verdict: GpuVerdictArtifact,
    *,
    config_sha256: str,
    tolerances_sha256: str,
    corpus_sha256: str,
    backbone_sha256: str,
) -> None:
    verdict.validate()
    if verdict.verdict != "PASS":
        raise GovernanceError("latest GPU identity verdict is not PASS")
    expected = {
        "config": config_sha256,
        "tolerances": tolerances_sha256,
        "corpus": corpus_sha256,
        "backbone": backbone_sha256,
    }
    actual = {
        "config": verdict.config_sha256,
        "tolerances": verdict.tolerances_sha256,
        "corpus": verdict.corpus_sha256,
        "backbone": verdict.backbone_sha256,
    }
    changed = [name for name in expected if expected[name] != actual[name]]
    if changed:
        raise GovernanceError(
            "latest PASS is stale; changed identity source(s): " + ", ".join(changed)
        )


def verify_tolerance_governance(
    path: str | Path,
    *,
    tolerances_path: str | Path,
    repo_root: str | Path,
) -> None:
    governance = json.loads(Path(path).read_text(encoding="utf-8"))
    _strict(governance, {"schema_version", "records"}, "tolerance_governance")
    if governance["schema_version"] != 1 or not isinstance(governance["records"], list):
        raise GovernanceError("invalid tolerance governance schema")
    tolerance_hash = hashlib.sha256(Path(tolerances_path).read_bytes()).hexdigest()
    matches = [item for item in governance["records"] if item.get("tolerances_sha256") == tolerance_hash]
    if len(matches) != 1:
        raise GovernanceError("current tolerances hash has no unique governance record")
    record = matches[0]
    _strict(record, {"tolerances_sha256", "report", "report_sha256", "adr"}, "governance.record")
    root = Path(repo_root)
    report = root / record["report"]
    adr = root / record["adr"]
    if not report.is_file() or not adr.is_file():
        raise GovernanceError("tolerance governance report or ADR is missing")
    if hashlib.sha256(report.read_bytes()).hexdigest() != record["report_sha256"]:
        raise GovernanceError("tolerance governance report hash changed")


def _strict(value: Any, expected: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise GovernanceError(f"{path} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise GovernanceError(f"{path}: missing={missing}, unknown={unknown}")
