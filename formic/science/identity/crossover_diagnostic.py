"""Pure design and CPU-only analysis for the balanced crossover diagnostic.

The module does not load or execute a model.  GPU execution remains isolated in
``schedule_diagnostic``; this module retains only explicitly observed CPU logits
and guarantees that canonical JSON checkpoints are tensor-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from formic.science.identity.artifacts import (
    ArtifactError,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
)
from formic.science.identity.metrics import compare_logits


CALENDARS = ("abba", "baab")
PAIR_NAMES = (
    "reference_reference",
    "runner_runner",
    "reference_runner",
    "runner_reference",
)
PAIR_ENDPOINTS = {
    "reference_reference": ("reference", "reference"),
    "runner_runner": ("runner", "runner"),
    "reference_runner": ("reference", "runner"),
    "runner_reference": ("runner", "reference"),
}
MEASURED_ROUNDS = 8
MEASURED_REPETITIONS = 3
FORWARDS_PER_PAIR = 16
WARMUP_DIAGNOSTIC_FORWARDS = 96
FORWARDS_PER_MEASURED_ROUND = 384
EXPECTED_SAME_SLOT_CONTRASTS = 1_536
EXPECTED_INVERSION_CHECKS = 768


class CrossoverBlocked(RuntimeError):
    """The evidence is invalid or insufficient and must not be mixed."""


@dataclass(frozen=True)
class BalancedConfiguration:
    round: int
    phase_index: int
    calendar: str
    pair: str
    configuration_ordinal: int

    @property
    def checkpoint_id(self) -> str:
        return (
            f"round_{self.round}__configuration_{self.configuration_ordinal}"
            f"__{self.calendar}__{self.pair}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def round_configurations(round_index: int) -> tuple[BalancedConfiguration, ...]:
    """Return one round's two phases and cyclic Latin-square pair rotations."""
    if not 0 <= round_index < MEASURED_ROUNDS:
        raise ValueError(f"round must be in 0..7, got {round_index}")
    phase_order = CALENDARS if round_index < 4 else tuple(reversed(CALENDARS))
    rotation = round_index % len(PAIR_NAMES)
    rotated_pairs = PAIR_NAMES[rotation:] + PAIR_NAMES[:rotation]
    return tuple(
        BalancedConfiguration(
            round=round_index,
            phase_index=phase_index,
            calendar=calendar,
            pair=pair,
            configuration_ordinal=phase_index * len(PAIR_NAMES) + pair_position,
        )
        for phase_index, calendar in enumerate(phase_order)
        for pair_position, pair in enumerate(rotated_pairs)
    )


def balanced_design() -> tuple[BalancedConfiguration, ...]:
    design = tuple(
        configuration
        for round_index in range(MEASURED_ROUNDS)
        for configuration in round_configurations(round_index)
    )
    validate_balanced_design(design)
    return design


def validate_balanced_design(
    design: Iterable[BalancedConfiguration],
) -> dict[str, Any]:
    items = tuple(design)
    if len(items) != MEASURED_ROUNDS * len(CALENDARS) * len(PAIR_NAMES):
        raise CrossoverBlocked("balanced design must contain exactly 64 configurations")
    coverage = {
        (calendar, pair): sorted(
            item.configuration_ordinal
            for item in items
            if item.calendar == calendar and item.pair == pair
        )
        for calendar in CALENDARS
        for pair in PAIR_NAMES
    }
    expected = list(range(8))
    failures = {
        f"{calendar}__{pair}": ordinals
        for (calendar, pair), ordinals in coverage.items()
        if ordinals != expected
    }
    round_failures = [
        round_index
        for round_index in range(MEASURED_ROUNDS)
        if sorted(
            item.configuration_ordinal for item in items if item.round == round_index
        )
        != expected
    ]
    if failures or round_failures:
        raise CrossoverBlocked(
            f"Latin-square coverage invalid: pairs={failures}, rounds={round_failures}"
        )
    return {
        "valid": True,
        "rounds": MEASURED_ROUNDS,
        "configurations_per_round": 8,
        "scheduled_occurrences": 64,
        "measured_repetitions_per_occurrence": MEASURED_REPETITIONS,
        "measured_pair_traces": 64 * MEASURED_REPETITIONS,
        "round_relative_global_forward_ordinal": {
            "formula": (
                "((configuration_ordinal * 3) + repetition) * 16 "
                "+ pair_local_forward_ordinal"
            ),
            "range_per_round": [0, 383],
            "scope": "round_relative_only",
        },
        "process_lifetime_diagnostic_forward_ordinal": {
            "formula": "96 + round * 384 + round_relative_global_forward_ordinal",
            "measured_range_per_attempt": [96, 3167],
            "counts_diagnostic_model_forwards_not_load_internals": True,
        },
        "matching_proof": {
            "matching_scope": "round_relative_balanced_crossover",
            "all_endpoint_treatments_cover_every_round_relative_slot": True,
            "latin_crossover_coverage_proved": True,
            "process_lifetime_ordinals_recorded": True,
            "process_lifetime_ordinals_equal_between_distinct_forwards": False,
            "limitation": (
                "Distinct endpoint-treatment forwards are matched by balanced round-relative calendar slot; their process-lifetime diagnostic ordinals are not equal."
            ),
            "causal_attribution": None,
        },
        "coverage": {
            f"{calendar}__{pair}": ordinals
            for (calendar, pair), ordinals in coverage.items()
        },
    }


def round_relative_global_forward_ordinal(
    configuration_ordinal: int,
    repetition: int,
    pair_local_forward_ordinal: int,
) -> int:
    if not 0 <= configuration_ordinal < 8:
        raise ValueError("configuration ordinal must be in 0..7")
    if not 0 <= pair_local_forward_ordinal < FORWARDS_PER_PAIR:
        raise ValueError("pair-local forward ordinal must be in 0..15")
    if not 0 <= repetition < MEASURED_REPETITIONS:
        raise ValueError("repetition must be in 0..2")
    return (
        (configuration_ordinal * MEASURED_REPETITIONS + repetition)
        * FORWARDS_PER_PAIR
        + pair_local_forward_ordinal
    )


def process_lifetime_diagnostic_forward_ordinal(
    round: int,
    configuration_ordinal: int,
    repetition: int,
    pair_local_forward_ordinal: int,
) -> int:
    """Return the exact measured forward position within one diagnostic attempt."""
    if not 0 <= round < MEASURED_ROUNDS:
        raise ValueError("round must be in 0..7")
    round_relative = round_relative_global_forward_ordinal(
        configuration_ordinal, repetition, pair_local_forward_ordinal
    )
    return (
        WARMUP_DIAGNOSTIC_FORWARDS
        + round * FORWARDS_PER_MEASURED_ROUND
        + round_relative
    )


@dataclass(frozen=True)
class CrossoverIdentity:
    protocol: str
    config_sha256: str
    corpus_sha256: str
    corpus_source_sha256: str
    git_commit: str
    backbone_sha256: str


class CrossoverWriter:
    """Atomic writer for mutable diagnostics and immutable config/round evidence."""

    def __init__(self, root: str | Path, identity: CrossoverIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.manifest_path = self.root / "manifest.json"
        self.configurations_dir = self.root / "configurations"
        self.rounds_dir = self.root / "rounds"
        self.diagnostics_dir = self.root / "diagnostics"
        self.configurations_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = self._read_manifest()
            if manifest["identity"] != asdict(identity):
                raise ArtifactError("resume identity differs from crossover manifest")
        else:
            atomic_write_json(
                self.manifest_path,
                {
                    "schema_version": 1,
                    "identity": asdict(identity),
                    "completed_configurations": {},
                    "completed_rounds": {},
                },
            )

    def completed_rounds(self) -> frozenset[int]:
        return frozenset(
            int(value) for value in self._read_manifest()["completed_rounds"]
        )

    def write_diagnostic(self, item_id: str, payload: dict[str, Any]) -> Path:
        _safe_id(item_id)
        _assert_tensor_free(payload)
        target = self.diagnostics_dir / f"{item_id}.json"
        atomic_write_json(target, payload)
        return target

    def write_configuration(self, item_id: str, payload: dict[str, Any]) -> Path:
        return self._write_immutable(
            "completed_configurations", self.configurations_dir, item_id, payload
        )

    def write_round(self, round_index: int, payload: dict[str, Any]) -> Path:
        if not 0 <= round_index < MEASURED_ROUNDS:
            raise ArtifactError("round checkpoint index must be in 0..7")
        return self._write_immutable(
            "completed_rounds", self.rounds_dir, str(round_index), payload
        )

    def validate(self) -> None:
        manifest = self._read_manifest()
        for key, directory in (
            ("completed_configurations", self.configurations_dir),
            ("completed_rounds", self.rounds_dir),
        ):
            for item_id, digest in manifest[key].items():
                path = directory / f"{item_id}.json"
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ArtifactError(f"crossover checkpoint invalid: {key}/{item_id}")

    def _write_immutable(
        self,
        manifest_key: str,
        directory: Path,
        item_id: str,
        payload: dict[str, Any],
    ) -> Path:
        _safe_id(item_id)
        _assert_tensor_free(payload)
        manifest = self._read_manifest()
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        target = directory / f"{item_id}.json"
        existing = manifest[manifest_key].get(item_id)
        if existing is not None:
            if existing != digest:
                raise ArtifactError(
                    f"completed crossover checkpoint changed during replay: {item_id}"
                )
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ArtifactError(f"completed crossover checkpoint is corrupt: {item_id}")
            return target
        atomic_write_json(target, payload)
        manifest[manifest_key][item_id] = digest
        atomic_write_json(self.manifest_path, manifest)
        return target

    def _read_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid crossover manifest") from exc
        expected = {
            "schema_version",
            "identity",
            "completed_configurations",
            "completed_rounds",
        }
        if set(value) != expected or value["schema_version"] != 1:
            raise ArtifactError("invalid crossover manifest schema")
        if not isinstance(value["completed_configurations"], dict) or not isinstance(
            value["completed_rounds"], dict
        ):
            raise ArtifactError("invalid crossover manifest checkpoints")
        return value


class AttemptMemoryWriter:
    """Append attempt-prefixed CUDA observations without replacing prior evidence."""

    def __init__(self, path: str | Path, attempt_id: str) -> None:
        self.path = Path(path)
        self.attempt_id = attempt_id
        self.measurements: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError("existing crossover CUDA memory evidence is invalid") from exc
            if set(value) != {"schema_version", "measurements"} or value["schema_version"] != 1:
                raise ArtifactError("existing crossover CUDA memory evidence has invalid schema")
            if not isinstance(value["measurements"], list):
                raise ArtifactError("existing crossover CUDA memory measurements are invalid")
            self.measurements = list(value["measurements"])

    def record(self, label: str, model: Any | None = None) -> dict[str, Any]:
        from formic.science.identity.memory import cuda_memory_measurement

        measurement = cuda_memory_measurement(f"{self.attempt_id}:{label}", model)
        self.measurements.append(measurement)
        atomic_write_json(
            self.path,
            {"schema_version": 1, "measurements": self.measurements},
        )
        return measurement


def prepare_attempt_metadata(
    path: str | Path,
    base: dict[str, Any],
    attempt: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Append one run attempt while preserving prior metadata verbatim."""
    target = Path(path)
    attempts: list[dict[str, Any]] = []
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("existing crossover run metadata is invalid") from exc
        if "attempts" in existing:
            attempts = existing.pop("attempts")
            if not isinstance(attempts, list):
                raise ArtifactError("existing crossover attempts are invalid")
        else:
            legacy_attempt = {
                key: existing.pop(key)
                for key in ("resume", "placement_override", "environment")
                if key in existing
            }
            if legacy_attempt:
                attempts = [{"attempt_id": "attempt_000", **legacy_attempt}]
        if existing != base:
            raise ArtifactError("resume run metadata identity or protocol differs")
    attempt_id = f"attempt_{len(attempts):03d}"
    attempts.append({"attempt_id": attempt_id, **attempt})
    result = {**base, "attempts": attempts}
    atomic_write_json(target, result)
    return result, attempt_id


def assert_resumable_terminal(path: str | Path) -> None:
    """Refuse resume after successful completion; allow absent or failed attempts."""
    target = Path(path)
    if not target.exists():
        return
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("existing crossover terminal artifact is invalid") from exc
    status = value.get("status") if isinstance(value, dict) else None
    if status == "COMPLETE":
        raise ArtifactError("refusing to resume an already COMPLETE crossover diagnostic")
    if status not in ("FAIL", "BLOCKED"):
        raise ArtifactError(f"existing crossover terminal status is not resumable: {status!r}")


class CPULogitBank:
    """Non-serialised CPU logits indexed by balanced measurement slot."""

    _KEY_FIELDS = (
        "calendar",
        "configuration_ordinal",
        "pair",
        "repetition",
        "side",
        "decode_step",
    )

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], tuple[dict[str, Any], torch.Tensor]] = {}

    def add(self, metadata: dict[str, Any], logits: torch.Tensor) -> None:
        missing = [field for field in self._KEY_FIELDS if field not in metadata]
        if missing:
            raise CrossoverBlocked(f"CPU logit metadata missing fields: {missing}")
        if logits.device.type != "cpu" or logits.requires_grad or logits.ndim != 1:
            raise CrossoverBlocked("crossover bank accepts detached one-dimensional CPU logits only")
        key = tuple(metadata[field] for field in self._KEY_FIELDS)
        if key in self._entries:
            raise CrossoverBlocked(f"duplicate CPU logit slot: {key}")
        self._entries[key] = (dict(metadata), logits.detach().contiguous())

    def get(
        self,
        calendar: str,
        configuration_ordinal: int,
        pair: str,
        repetition: int,
        side: str,
        step: int,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        key = (calendar, configuration_ordinal, pair, repetition, side, step)
        try:
            return self._entries[key]
        except KeyError as exc:
            raise CrossoverBlocked(f"missing CPU logit slot: {key}") from exc

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def compare_matched_logits(
    reference_metadata: dict[str, Any],
    reference_logits: torch.Tensor,
    candidate_metadata: dict[str, Any],
    candidate_logits: torch.Tensor,
    *,
    equal_fields: tuple[str, ...] = (
        "calendar",
        "configuration_ordinal",
        "decode_step",
        "side",
        "pair_local_forward_ordinal",
        "round_relative_global_forward_ordinal",
        "repetition",
    ),
) -> dict[str, Any]:
    """Compare CPU logits only after proving their declared ordinal slot matches."""
    _validate_ordinal_metadata(reference_metadata)
    _validate_ordinal_metadata(candidate_metadata)
    mismatches = {
        field: (reference_metadata.get(field), candidate_metadata.get(field))
        for field in equal_fields
        if reference_metadata.get(field) != candidate_metadata.get(field)
    }
    if mismatches:
        raise CrossoverBlocked(f"logit ordinal metadata mismatch: {mismatches}")
    if reference_logits.device.type != "cpu" or candidate_logits.device.type != "cpu":
        raise CrossoverBlocked("numeric matching is CPU-only")
    metric = compare_logits(reference_logits, candidate_logits).to_dict()
    tensor = metric["tensor"]
    return {
        "exact": tensor["exact"],
        "max_abs_delta": tensor["max_abs_delta"],
        "kl_next_token": metric["kl_next_token"],
        "reference_top1": metric["reference_top1"],
        "candidate_top1": metric["candidate_top1"],
        "top1_agreement": metric["top1_agreement"],
        "first_coordinate": tensor["first_coordinate"],
        "reference_value": tensor["reference_value"],
        "candidate_value": tensor["candidate_value"],
    }


def build_matched_slot_evidence(
    reference_metadata: dict[str, Any],
    reference_logits: torch.Tensor,
    candidate_metadata: dict[str, Any],
    candidate_logits: torch.Tensor,
    *,
    equal_fields: tuple[str, ...] = (
        "calendar",
        "configuration_ordinal",
        "decode_step",
        "side",
        "pair_local_forward_ordinal",
        "round_relative_global_forward_ordinal",
        "repetition",
    ),
) -> dict[str, Any]:
    """Build transparent evidence for one balanced round-relative slot match."""
    metric = compare_matched_logits(
        reference_metadata,
        reference_logits,
        candidate_metadata,
        candidate_logits,
        equal_fields=equal_fields,
    )
    reference_process = reference_metadata[
        "process_lifetime_diagnostic_forward_ordinal"
    ]
    candidate_process = candidate_metadata[
        "process_lifetime_diagnostic_forward_ordinal"
    ]
    if reference_process == candidate_process:
        raise CrossoverBlocked(
            "distinct balanced treatment forwards cannot share a process-lifetime diagnostic ordinal"
        )
    return {
        "matching_scope": "round_relative_balanced_crossover",
        "round_relative_calendar_slot_matched": True,
        "round_relative_global_forward_ordinal": reference_metadata[
            "round_relative_global_forward_ordinal"
        ],
        "reference_round": reference_metadata["round"],
        "candidate_round": candidate_metadata["round"],
        "reference_process_lifetime_diagnostic_forward_ordinal": reference_process,
        "candidate_process_lifetime_diagnostic_forward_ordinal": candidate_process,
        "process_lifetime_diagnostic_forward_ordinals_matched": False,
        "metric": metric,
    }


_SAME_SLOT_CONTRASTS = (
    ("left_companion_reference", "reference_reference", "runner_reference", "left"),
    ("left_companion_runner", "reference_runner", "runner_runner", "left"),
    ("right_companion_reference", "reference_reference", "reference_runner", "right"),
    ("right_companion_runner", "runner_reference", "runner_runner", "right"),
)


def build_same_slot_contrasts(bank: CPULogitBank) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for calendar in CALENDARS:
        for configuration_ordinal in range(8):
            for repetition in range(MEASURED_REPETITIONS):
                for step in range(8):
                    for contrast, reference_pair, candidate_pair, side in _SAME_SLOT_CONTRASTS:
                        reference_meta, reference_logits = bank.get(
                            calendar, configuration_ordinal, reference_pair, repetition, side, step
                        )
                        candidate_meta, candidate_logits = bank.get(
                            calendar, configuration_ordinal, candidate_pair, repetition, side, step
                        )
                        _expect_endpoint(reference_meta, "reference")
                        _expect_endpoint(candidate_meta, "runner")
                        results.append(
                            {
                                "contrast": contrast,
                                "calendar": calendar,
                                "configuration_ordinal": configuration_ordinal,
                                "repetition": repetition,
                                "step": step,
                                "side": side,
                                "reference_pair": reference_pair,
                                "candidate_pair": candidate_pair,
                                **build_matched_slot_evidence(
                                    reference_meta,
                                    reference_logits,
                                    candidate_meta,
                                    candidate_logits,
                                ),
                            }
                        )
    return results


def build_inversion_checks(bank: CPULogitBank) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    inversion_pairs = (
        ("reference_endpoint", "reference_runner", "left", "runner_reference", "right", "reference"),
        ("runner_endpoint", "reference_runner", "right", "runner_reference", "left", "runner"),
    )
    equal_fields = (
        "configuration_ordinal",
        "decode_step",
        "pair_local_forward_ordinal",
        "round_relative_global_forward_ordinal",
        "repetition",
        "endpoint",
    )
    for source_calendar, candidate_calendar in (("abba", "baab"), ("baab", "abba")):
        for configuration_ordinal in range(8):
            for repetition in range(MEASURED_REPETITIONS):
                for step in range(8):
                    for name, ref_pair, ref_side, cand_pair, cand_side, endpoint in inversion_pairs:
                        reference_meta, reference_logits = bank.get(
                            source_calendar,
                            configuration_ordinal,
                            ref_pair,
                            repetition,
                            ref_side,
                            step,
                        )
                        candidate_meta, candidate_logits = bank.get(
                            candidate_calendar,
                            configuration_ordinal,
                            cand_pair,
                            repetition,
                            cand_side,
                            step,
                        )
                        _expect_endpoint(reference_meta, endpoint)
                        _expect_endpoint(candidate_meta, endpoint)
                        results.append(
                            {
                                "inversion": name,
                                "source_calendar": source_calendar,
                                "candidate_calendar": candidate_calendar,
                                "configuration_ordinal": configuration_ordinal,
                                "repetition": repetition,
                                "step": step,
                                **build_matched_slot_evidence(
                                    reference_meta,
                                    reference_logits,
                                    candidate_meta,
                                    candidate_logits,
                                    equal_fields=equal_fields,
                                ),
                            }
                        )
    return results


def stability_against_prior(
    current_metadata: dict[str, Any],
    current_logits: torch.Tensor,
    prior_metadata: dict[str, Any] | None,
    prior_logits: torch.Tensor | None,
) -> dict[str, Any] | None:
    if prior_metadata is None or prior_logits is None:
        return None
    metric = compare_matched_logits(
        prior_metadata,
        prior_logits,
        current_metadata,
        current_logits,
        equal_fields=(
            "calendar",
            "configuration_ordinal",
            "pair",
            "decode_step",
            "side",
            "pair_local_forward_ordinal",
            "endpoint",
        ),
    )
    return {
        "exact": metric["exact"],
        "max_abs_delta": metric["max_abs_delta"],
        "kl_next_token": metric["kl_next_token"],
        "first_divergence": metric["first_coordinate"],
        "prior_value": metric["reference_value"],
        "current_value": metric["candidate_value"],
    }


def build_ordinal_position_observations(
    configurations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Associate output hashes with round-relative configuration ordinal only."""
    buckets: dict[tuple[str, str, int, str, int], dict[int, list[str]]] = {}
    for configuration in configurations:
        calendar = configuration.get("calendar")
        pair = configuration.get("pair")
        ordinal = configuration.get("configuration_ordinal")
        for repetition_payload in configuration.get("repetitions", ()):
            repetition = repetition_payload.get("repetition")
            for slot in repetition_payload.get("slots", ()):
                key = (
                    calendar,
                    pair,
                    repetition,
                    slot.get("side"),
                    slot.get("decode_step"),
                )
                buckets.setdefault(key, {}).setdefault(ordinal, []).append(slot.get("sha256"))

    observations: list[dict[str, Any]] = []
    expected_ordinals = list(range(8))
    for calendar in CALENDARS:
        for pair in PAIR_NAMES:
            for repetition in range(MEASURED_REPETITIONS):
                for side in ("left", "right"):
                    for step in range(8):
                        key = (calendar, pair, repetition, side, step)
                        values = buckets.get(key, {})
                        exact_coverage = (
                            sorted(values) == expected_ordinals
                            and all(len(items) == 1 and isinstance(items[0], str) for items in values.values())
                        )
                        hashes = [values[ordinal][0] for ordinal in expected_ordinals] if exact_coverage else []
                        first_difference = None
                        if exact_coverage:
                            first = next(
                                (ordinal for ordinal in range(1, 8) if hashes[ordinal] != hashes[0]),
                                None,
                            )
                            if first is not None:
                                first_difference = {
                                    "baseline_ordinal": 0,
                                    "baseline_sha256": hashes[0],
                                    "differing_ordinal": first,
                                    "differing_sha256": hashes[first],
                                }
                        observations.append(
                            {
                                "calendar": calendar,
                                "pair": pair,
                                "repetition": repetition,
                                "side": side,
                                "decode_step": step,
                                "expected_configuration_ordinals": expected_ordinals,
                                "observed_configuration_ordinals": sorted(values),
                                "exact_coverage": exact_coverage,
                                "hashes_by_configuration_ordinal": [
                                    {
                                        "configuration_ordinal": ordinal,
                                        "sha256": values[ordinal][0],
                                    }
                                    for ordinal in expected_ordinals
                                    if ordinal in values and len(values[ordinal]) == 1
                                ],
                                "hashes_change_with_ordinal": (
                                    len(set(hashes)) > 1 if exact_coverage else None
                                ),
                                "first_differing_ordinal_hash": first_difference,
                                "interpretation": (
                                    "Observed association with round-relative ordinal only; no cause is attributed."
                                ),
                            }
                        )
    coverage_count = sum(item["exact_coverage"] for item in observations)
    changed = sum(item["hashes_change_with_ordinal"] is True for item in observations)
    return {
        "schema_version": 1,
        "kind": "round-relative ordinal-position hash observations",
        "causal_attribution": None,
        "observations": observations,
        "aggregate": {
            "expected_groups": 384,
            "observed_groups": len(observations),
            "exact_coverage_groups": coverage_count,
            "incomplete_coverage_groups": len(observations) - coverage_count,
            "groups_with_hash_change_by_ordinal": changed,
            "groups_without_hash_change_by_ordinal": sum(
                item["hashes_change_with_ordinal"] is False for item in observations
            ),
            "any_observed_hash_change_by_ordinal": changed > 0,
            "interpretation": (
                "Hash changes are observed associations with round-relative ordinal position, not causal attribution."
            ),
        },
    }


def build_analysis(
    *,
    configurations: Iterable[dict[str, Any]],
    same_slot_contrasts: Iterable[dict[str, Any]],
    inversion_checks: Iterable[dict[str, Any]],
    design_validation: dict[str, Any],
) -> dict[str, Any]:
    configurations = list(configurations)
    contrasts = list(same_slot_contrasts)
    inversions = list(inversion_checks)
    ordinal_positions = build_ordinal_position_observations(configurations)
    expected_configurations = {
        (item.round, item.configuration_ordinal, item.calendar, item.pair)
        for item in balanced_design()
    }
    observed_configurations = {
        (
            item.get("round"),
            item.get("configuration_ordinal"),
            item.get("calendar"),
            item.get("pair"),
        )
        for item in configurations
    }
    complete = (
        len(configurations) == 64
        and observed_configurations == expected_configurations
        and all(item.get("status") == "COMPLETE" for item in configurations)
        and all(
            [item.get("repetition") for item in configuration.get("repetitions", ())]
            == list(range(MEASURED_REPETITIONS))
            and all(
                len(item.get("slots", ())) == 16
                for item in configuration.get("repetitions", ())
            )
            for configuration in configurations
        )
    )
    controls = [
        item
        for item in configurations
        if item.get("pair") in ("reference_reference", "runner_runner")
    ]
    controls_evidence_complete = complete and len(controls) == 32 and all(
        isinstance(item.get("all_stability_exact"), bool) for item in controls
    )
    controls_stable = controls_evidence_complete and all(
        item.get("all_stability_exact") is True for item in controls
    )
    last_two_evidence_complete = complete and all(
        isinstance(item.get("all_last_two_slots_exact"), bool)
        for item in configurations
    )
    last_two_exact = last_two_evidence_complete and all(
        item.get("all_last_two_slots_exact") is True for item in configurations
    )
    same_slot_summary = _metric_summary(contrasts, EXPECTED_SAME_SLOT_CONTRASTS)
    inversion_summary = _metric_summary(inversions, EXPECTED_INVERSION_CHECKS)
    same_slot_evidence_complete = (
        complete
        and same_slot_summary["observed_count"] == EXPECTED_SAME_SLOT_CONTRASTS
    )
    inversion_evidence_complete = (
        complete
        and inversion_summary["observed_count"] == EXPECTED_INVERSION_CHECKS
    )
    same_slot_exact = (
        same_slot_evidence_complete
        and same_slot_summary["nonexact_count"] == 0
    )
    matched_contrast_stability = _matched_contrast_stability(contrasts)
    matched_contrasts_last_two_exact = (
        same_slot_evidence_complete
        and matched_contrast_stability["complete"]
        and matched_contrast_stability["last_two_exact"]
    )
    inversions_exact = (
        inversion_evidence_complete
        and inversion_summary["nonexact_count"] == 0
    )
    design_valid = design_validation.get("valid") is True
    ordinal_coverage_complete = (
        ordinal_positions["aggregate"]["exact_coverage_groups"] == 384
    )
    method_complete = all(
        (
            complete,
            design_valid,
            ordinal_coverage_complete,
            same_slot_summary["observed_count"] == EXPECTED_SAME_SLOT_CONTRASTS,
            inversion_summary["observed_count"] == EXPECTED_INVERSION_CHECKS,
        )
    )
    ready = all(
        (
            complete,
            design_valid,
            same_slot_exact,
            matched_contrasts_last_two_exact,
            ordinal_coverage_complete,
        )
    )
    command_after_adaptation = (
        "python scripts/step2_a40_campaign.py --config configs/default.yaml "
        "--run-id <RUN_ID> --sampled-continuation-seed 0"
        if ready
        else None
    )
    pair_summaries = _pair_result_summaries(configurations, ordinal_positions)
    questions = [
        {
            "question_id": "a_outputs_change_by_ordinal_position",
            "question_fr": "Les sorties changent-elles selon la position ordinale ?",
            "answer": (
                ordinal_positions["aggregate"]["any_observed_hash_change_by_ordinal"]
                if ordinal_coverage_complete
                else None
            ),
            "evidence": ordinal_positions["aggregate"],
            "interpretation": "Association observee uniquement; aucune cause n'est attribuee.",
        },
        {
            "question_id": "b_outputs_change_by_endpoint_at_matched_ordinal",
            "question_fr": "Les sorties changent-elles selon l'endpoint a ordinal apparie ?",
            "answer": (
                same_slot_summary["nonexact_count"] > 0
                if same_slot_evidence_complete
                else None
            ),
            "evidence": same_slot_summary,
        },
        {
            "question_id": "c_rr_nn_controls_stable",
            "question_fr": "Les controles reference/reference et runner/runner sont-ils stables ?",
            "answer": controls_stable if controls_evidence_complete else None,
            "configurations": len(controls),
        },
        {
            "question_id": "d_rn_nr_inversion_coherent",
            "question_fr": "L'inversion RN/NR est-elle coherente ?",
            "answer": inversions_exact if inversion_evidence_complete else None,
            "evidence": inversion_summary,
        },
        {
            "question_id": "e_reference_runner_exact_after_slot_balancing",
            "question_fr": "Reference et runner sont-ils exacts apres equilibrage des slots ?",
            "answer": same_slot_exact if same_slot_evidence_complete else None,
            "evidence": same_slot_summary,
        },
        {
            "question_id": "f_last_two_repetitions_exact_per_slot",
            "question_fr": "Les deux dernieres repetitions sont-elles exactes pour chaque slot ?",
            "answer": last_two_exact if last_two_evidence_complete else None,
            "configurations": len(configurations),
        },
    ]
    return {
        "schema_version": 1,
        "kind": "balanced SPEC-02 crossover diagnostic analysis only",
        "status": "COMPLETE" if method_complete else "BLOCKED",
        "not_an_identity_verdict": True,
        "readiness": {
            "status": "READY" if ready else "BLOCKED",
            "ready_for_full_spec_02_campaign": ready,
            "official_launcher_requires_calendar_adaptation": False,
            "official_launcher_usable_as_is": ready,
            "currently_runnable": ready,
            "official_command_recommendation": command_after_adaptation,
            "command_template_after_code_adaptation": command_after_adaptation,
            "launcher_note": (
                "The official launcher includes the four-slot Latin ABBA endpoint gate and cross-path calibration protocol."
                if ready
                else "No official campaign command is recommended because readiness is blocked."
            ),
            "tolerances_changed": False,
            "cause_attribution": None,
        },
        "checks": {
            "all_configurations_complete": complete,
            "balanced_design_valid": design_valid,
            "same_slot_contrasts_exact": same_slot_exact,
            "matched_contrasts_last_two_exact": matched_contrasts_last_two_exact,
            "controls_stable": controls_stable,
            "inversion_checks_coherent_and_exact": inversions_exact,
            "all_last_two_slot_checks_exact": last_two_exact,
            "ordinal_position_coverage_complete": ordinal_coverage_complete,
        },
        "counts": {
            "configurations": len(configurations),
            "same_slot_contrasts": len(contrasts),
            "inversion_checks": len(inversions),
            "measured_pair_traces": sum(
                len(item.get("repetitions", ())) for item in configurations
            ),
        },
        "same_slot_metric_summary": same_slot_summary,
        "matched_contrast_stability": matched_contrast_stability,
        "inversion_metric_summary": inversion_summary,
        "ordinal_position_summary": ordinal_positions["aggregate"],
        "pair_result_summaries": pair_summaries,
        "design_supporting_evidence": design_validation,
        "matching_proof_and_limitations": {
            "matching_scope": "round_relative_balanced_crossover",
            "all_endpoint_treatments_cover_every_round_relative_slot": design_valid,
            "round_relative_calendar_slot_matching_used": (
                same_slot_evidence_complete and inversion_evidence_complete
            ),
            "process_lifetime_diagnostic_ordinals_recorded": complete,
            "process_lifetime_diagnostic_ordinals_expected_equal_between_distinct_forwards": False,
            "limitation": (
                "Contrasts match Latin-balanced round-relative calendar slots across rounds. Distinct forwards have distinct process-lifetime diagnostic ordinals, which are recorded but not matched."
            ),
            "causal_attribution": None,
        },
        "nonblocking_ordinal_diagnostics": {
            "raw_controls_stable": controls_stable,
            "raw_inversions_exact": inversions_exact,
            "raw_last_two_slots_exact": last_two_exact,
            "reason": (
                "These comparisons place distinct traces at distinct process-lifetime "
                "ordinals. ADR-0004 requires the matched endpoint contrast, not raw "
                "cross-ordinal fingerprints, to carry the wrapper identity verdict."
            ),
            "causal_attribution": None,
        },
        "questions_fr": questions,
    }


def _matched_contrast_stability(contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert stability of the measured endpoint contrast, not raw ordinals."""
    grouped: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for item in contrasts:
        key = (
            item.get("contrast"),
            item.get("calendar"),
            item.get("configuration_ordinal"),
            item.get("step"),
            item.get("side"),
        )
        metric = item.get("metric", {})
        grouped.setdefault(key, {})[int(item.get("repetition", -1))] = {
            field: metric.get(field)
            for field in (
                "exact",
                "max_abs_delta",
                "kl_next_token",
                "top1_agreement",
                "first_coordinate",
                "reference_value",
                "candidate_value",
            )
        }
    expected_groups = EXPECTED_SAME_SLOT_CONTRASTS // MEASURED_REPETITIONS
    failures: list[dict[str, Any]] = []
    complete = len(grouped) == expected_groups
    for key, repetitions in sorted(grouped.items(), key=lambda item: str(item[0])):
        if sorted(repetitions) != list(range(MEASURED_REPETITIONS)):
            complete = False
            failures.append({"key": list(key), "reason": "missing_repetition"})
            continue
        if canonical_json_bytes(repetitions[1]) != canonical_json_bytes(repetitions[2]):
            failures.append({
                "key": list(key),
                "reason": "last_two_contrast_metrics_differ",
                "repetition_1": repetitions[1],
                "repetition_2": repetitions[2],
            })
    return {
        "assertion": "last_two_matched_endpoint_contrasts_exact",
        "expected_groups": expected_groups,
        "observed_groups": len(grouped),
        "complete": complete,
        "last_two_exact": complete and not failures,
        "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
    }


def _metric_summary(observations: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    exact = [item for item in observations if item.get("metric", {}).get("exact") is True]
    nonexact = [item for item in observations if item.get("metric", {}).get("exact") is not True]
    first_nonexact = None
    if nonexact:
        item = nonexact[0]
        metric = item.get("metric", {})
        first_nonexact = {
            **{key: value for key, value in item.items() if key != "metric"},
            "first_coordinate": metric.get("first_coordinate"),
            "reference_value": metric.get("reference_value", metric.get("left_value")),
            "candidate_value": metric.get("candidate_value", metric.get("right_value")),
            "max_abs_delta": metric.get("max_abs_delta"),
            "kl_next_token": metric.get("kl_next_token"),
        }
    return {
        "expected_count": expected,
        "observed_count": len(observations),
        "exact_count": len(exact),
        "nonexact_count": len(nonexact),
        "max_max_abs_delta": max(
            (float(item.get("metric", {}).get("max_abs_delta", 0.0)) for item in observations),
            default=None,
        ),
        "max_kl_next_token": max(
            (float(item.get("metric", {}).get("kl_next_token", 0.0)) for item in observations),
            default=None,
        ),
        "first_nonexact": first_nonexact,
    }


def _pair_result_summaries(
    configurations: list[dict[str, Any]],
    ordinal_positions: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for pair in PAIR_NAMES:
        pair_configs = [item for item in configurations if item.get("pair") == pair]
        comparisons = []
        for configuration in pair_configs:
            for repetition in configuration.get("repetitions", ()):
                comparisons.extend(
                    {"metric": slot["pair_comparison"]}
                    for slot in repetition.get("slots", ())
                    if slot.get("side") == "left" and "pair_comparison" in slot
                )
        position_groups = [
            item
            for item in ordinal_positions["observations"]
            if item["pair"] == pair
        ]
        result.append(
            {
                "pair": pair,
                "expected_configurations": 16,
                "observed_configurations": len(pair_configs),
                "expected_pair_traces": 48,
                "observed_pair_traces": sum(
                    len(item.get("repetitions", ())) for item in pair_configs
                ),
                "all_stability_exact": bool(pair_configs)
                and all(item.get("all_stability_exact") is True for item in pair_configs),
                "all_last_two_slots_exact": bool(pair_configs)
                and all(item.get("all_last_two_slots_exact") is True for item in pair_configs),
                "pair_comparison_metrics": _metric_summary(comparisons, 384),
                "ordinal_position_groups": {
                    "expected": 96,
                    "observed": len(position_groups),
                    "exact_coverage": sum(item["exact_coverage"] for item in position_groups),
                    "hash_changes": sum(
                        item["hashes_change_with_ordinal"] is True
                        for item in position_groups
                    ),
                },
            }
        )
    return result


def iter_slot_records(result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for step in result["steps"]:
        yield step["left"]
        yield step["right"]


def _expect_endpoint(metadata: dict[str, Any], expected: str) -> None:
    if metadata.get("endpoint") != expected:
        raise CrossoverBlocked(
            f"endpoint metadata mismatch: expected {expected}, got {metadata.get('endpoint')}"
        )


def _validate_ordinal_metadata(metadata: dict[str, Any]) -> None:
    try:
        round_index = metadata["round"]
        configuration = metadata["configuration_ordinal"]
        repetition = metadata["repetition"]
        local = metadata["pair_local_forward_ordinal"]
        global_ordinal = metadata["round_relative_global_forward_ordinal"]
        process_ordinal = metadata["process_lifetime_diagnostic_forward_ordinal"]
        step = metadata["decode_step"]
        within = metadata["within_step_ordinal"]
    except KeyError as exc:
        raise CrossoverBlocked(f"logit ordinal metadata missing: {exc.args[0]}") from exc
    if local != 2 * step + within or within not in (0, 1):
        raise CrossoverBlocked("pair-local forward ordinal is inconsistent with decode step")
    expected_global = round_relative_global_forward_ordinal(
        configuration, repetition, local
    )
    if global_ordinal != expected_global:
        raise CrossoverBlocked(
            "round-relative global forward ordinal is inconsistent with configuration"
        )
    expected_process = process_lifetime_diagnostic_forward_ordinal(
        round_index, configuration, repetition, local
    )
    if process_ordinal != expected_process:
        raise CrossoverBlocked(
            "process-lifetime diagnostic forward ordinal is inconsistent with round metadata"
        )


def _safe_id(item_id: str) -> None:
    if not item_id or any(character in item_id for character in "/\\"):
        raise ArtifactError("crossover checkpoint id must be filename-safe")


def _assert_tensor_free(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise ArtifactError("canonical crossover JSON cannot contain tensors")
    if isinstance(value, dict):
        for item in value.values():
            _assert_tensor_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_tensor_free(item)
