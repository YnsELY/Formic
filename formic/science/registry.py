"""Experiment registry.

Plan rule: *chaque run = config + commit + seeds + coût*. Every measurement that
will ever be quoted must be traceable to an ``EXP-...`` entry, otherwise it does
not exist. The registry is an append-only JSONL file plus a rendered Markdown
table; nothing is ever edited in place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = ["ExperimentRecord", "ExperimentRegistry"]

DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "experiments"


@dataclass
class ExperimentRecord:
    experiment_id: str
    title: str
    step: str
    status: str = "RUNNING"  # RUNNING | DONE | FAILED | ABANDONED
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    git_commit: str | None = None
    config_hash: str | None = None
    config_path: str | None = None
    seeds: tuple[int, ...] = ()
    artifacts: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["artifacts"] = list(self.artifacts)
        return data


class ExperimentRegistry:
    """Append-only registry stored at ``experiments/registry.jsonl``."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_REGISTRY_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / "registry.jsonl"
        self.markdown_path = self.directory / "REGISTRY.md"

    # -- reading -----------------------------------------------------------

    def records(self) -> list[ExperimentRecord]:
        if not self.jsonl_path.is_file():
            return []
        out: list[ExperimentRecord] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["seeds"] = tuple(payload.get("seeds", ()))
            payload["artifacts"] = tuple(payload.get("artifacts", ()))
            out.append(ExperimentRecord(**payload))
        return out

    def latest_by_id(self) -> dict[str, ExperimentRecord]:
        """Last state of each experiment id (append-only log, last write wins)."""
        latest: dict[str, ExperimentRecord] = {}
        for record in self.records():
            latest[record.experiment_id] = record
        return latest

    def next_id(self) -> str:
        numbers = [
            int(record.experiment_id.split("-")[-1])
            for record in self.records()
            if record.experiment_id.startswith("EXP-")
        ]
        return f"EXP-{max(numbers) + 1 if numbers else 1:04d}"

    # -- writing -----------------------------------------------------------

    def append(self, record: ExperimentRecord) -> ExperimentRecord:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        self.render_markdown()
        return record

    def start(
        self,
        title: str,
        step: str,
        *,
        config_hash: str | None = None,
        config_path: str | None = None,
        seeds: Iterable[int] = (),
        environment: dict[str, Any] | None = None,
        notes: str = "",
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        from formic.science.determinism import git_commit

        record = ExperimentRecord(
            experiment_id=experiment_id or self.next_id(),
            title=title,
            step=step,
            status="RUNNING",
            git_commit=git_commit(),
            config_hash=config_hash,
            config_path=config_path,
            seeds=tuple(seeds),
            environment=environment or {},
            notes=notes,
        )
        return self.append(record)

    def finish(
        self,
        record: ExperimentRecord,
        *,
        status: str = "DONE",
        metrics: dict[str, Any] | None = None,
        cost: dict[str, Any] | None = None,
        artifacts: Iterable[str] = (),
        notes: str | None = None,
    ) -> ExperimentRecord:
        record.status = status
        record.finished_at = datetime.now(timezone.utc).isoformat()
        if metrics:
            record.metrics.update(metrics)
        if cost:
            record.cost.update(cost)
        if artifacts:
            record.artifacts = tuple(record.artifacts) + tuple(artifacts)
        if notes is not None:
            record.notes = notes
        return self.append(record)

    # -- rendering ---------------------------------------------------------

    def render_markdown(self) -> Path:
        records = list(self.latest_by_id().values())
        records.sort(key=lambda r: r.experiment_id)
        lines = [
            "# Formic experiment registry",
            "",
            "Append-only. Generated from `registry.jsonl` - do not edit by hand.",
            "",
            "| ID | Step | Title | Status | Started | Config hash | Commit | Seeds | Artifacts |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for record in records:
            lines.append(
                "| {id} | {step} | {title} | {status} | {started} | {cfg} | {commit} | {seeds} | {artifacts} |".format(
                    id=record.experiment_id,
                    step=record.step,
                    title=record.title.replace("|", "/"),
                    status=record.status,
                    started=record.started_at[:19],
                    cfg=(record.config_hash or "-")[:12],
                    commit=(record.git_commit or "-")[:8],
                    seeds=",".join(str(s) for s in record.seeds) or "-",
                    artifacts=str(len(record.artifacts)),
                )
            )
        if not records:
            lines.append("| _(empty)_ | | | | | | | | |")
        lines.append("")
        self.markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return self.markdown_path
