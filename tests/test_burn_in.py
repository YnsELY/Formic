"""Measured-then-discarded burn-in (protocol v3).

Run a40-2026-08-26-r1 failed because the first ~2 measured pair traces after a
capture-free warmup sat in a different numerical realisation than every later
trace (the documented ADR-0004 first-execution effect).  These weight-free
tests replay that exact failure shape against the balanced gate and verify
that the burn-in absorbs it without weakening any blocking criterion.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from formic.backbone.groups import HybridGroupView
from formic.science.identity import balanced_gate
from formic.science.identity.balanced_gate import (
    run_alternating_noise_floor,
    run_balanced_logits_gate,
)
from formic.science.identity.campaign import _MeasurementSession
from formic.science.identity.campaign_plan import CampaignPath
from formic.science.identity.executor import Endpoint
from formic.science.identity.types import ExecutionMode, SamplingMode
from tests.toy_qwen import toy_model


def _endpoints() -> tuple[Endpoint, Endpoint]:
    reference = Endpoint("reference", SimpleNamespace(), None, False)
    runner = Endpoint("runner", SimpleNamespace(), None, True)
    return reference, runner


class _SwitchingRealizationPair:
    """Fake ``run_schedule_pair`` reproducing the measured transient.

    The first ``dirty_pairs`` captured pair traces of the process emit
    realisation A logits; every later captured trace emits realisation B.
    Capture-free warmups do not advance the counter — exactly the observed
    behaviour where the capture-free warmup left the first measured pair in
    the pre-transient realisation.
    """

    def __init__(self, dirty_pairs: int) -> None:
        self.captured_pairs = 0
        self.dirty_pairs = dirty_pairs

    def __call__(
        self,
        calendar,
        left,
        right,
        *,
        prompt_token_ids,
        forced_token_ids,
        capture,
        cpu_logits_observer=None,
        **_kwargs,
    ):
        if not capture:
            return None
        realization = 0 if self.captured_pairs < self.dirty_pairs else 1
        self.captured_pairs += 1
        steps = []
        for step in range(len(forced_token_ids)):
            records = {}
            for within_step, side in enumerate(("left", "right")):
                endpoint = left if side == "left" else right
                metadata = {
                    "calendar": calendar,
                    "endpoint": endpoint.name,
                    "side": side,
                    "decode_step": step,
                    "step": step,
                    "pair_local_forward_ordinal": 2 * step + within_step,
                    "within_step_ordinal": within_step,
                }
                logits = torch.tensor([float(realization), 1.0 + float(realization)])
                if cpu_logits_observer is not None:
                    cpu_logits_observer(dict(metadata), logits)
                records[side] = {
                    **metadata,
                    "sha256": f"realization-{realization}",
                    "top1": 1,
                }
            steps.append(
                {
                    "step": step,
                    "point": "logits",
                    "left": records["left"],
                    "right": records["right"],
                    "comparison": {
                        "max_abs_delta": 0.0,
                        "kl_next_token": 0.0,
                        "left_top1": 1,
                        "right_top1": 1,
                        "top1_agreement": True,
                        "exact": True,
                        "first_coordinate": None,
                        "left_value": None,
                        "right_value": None,
                    },
                }
            )
        return {
            "left_path_fingerprint": f"{left.name}-r{realization}",
            "right_path_fingerprint": f"{right.name}-r{realization}",
            "steps": steps,
            "cache_independence": {
                "cache_objects_distinct": True,
                "cache_storage_disjoint": True,
            },
            "autograd_disabled_all_forwards": True,
        }


HORIZON_8 = (5, 6, 7, 8, 9, 10, 11, 12)


def test_balanced_gate_without_burn_in_reproduces_the_a40_failure(monkeypatch):
    monkeypatch.setattr(
        balanced_gate, "run_schedule_pair", _SwitchingRealizationPair(dirty_pairs=2)
    )
    reference, runner = _endpoints()
    with pytest.raises(RuntimeError, match="balanced endpoint identity comparison diverged"):
        run_balanced_logits_gate(
            reference,
            runner,
            prompt_token_ids=(1, 2, 3, 4),
            forced_token_ids=HORIZON_8,
            repetitions=2,
            warmup_pair_traces=6,
            burn_in_pair_traces=0,
        )


def test_balanced_gate_burn_in_absorbs_the_measured_transient(monkeypatch):
    fake = _SwitchingRealizationPair(dirty_pairs=2)
    monkeypatch.setattr(balanced_gate, "run_schedule_pair", fake)
    reference, runner = _endpoints()
    payload = run_balanced_logits_gate(
        reference,
        runner,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=HORIZON_8,
        repetitions=2,
        warmup_pair_traces=6,
        burn_in_pair_traces=4,
    )

    assert payload["matched_endpoint_exact"] is True
    assert payload["matched_contrast_last_two_exact"] is True
    assert payload["protocol"] == "SPEC-02-balanced-abba-latin4-h8-v3"
    burn_in = payload["burn_in"]
    assert burn_in["executed"] is True
    assert burn_in["executed_pair_traces"] == 4
    assert burn_in["excluded_from_blocking_criteria"] is True
    assert [item["pair"] for item in burn_in["pair_results"]] == [
        "reference_reference",
        "runner_runner",
        "reference_runner",
        "runner_reference",
    ]
    # The burn-in ran on the measured path: it consumed the dirty window.
    assert burn_in["pair_results"][0]["left_path_fingerprint"] == "reference-r0"
    assert burn_in["pair_results"][2]["left_path_fingerprint"] == "reference-r1"
    # Measured process ordinals start after warmup + burn-in.
    measured_ordinals = [
        item["reference_process_lifetime_diagnostic_forward_ordinal"]
        for item in payload["matched_contrasts"]
    ] + [
        item["candidate_process_lifetime_diagnostic_forward_ordinal"]
        for item in payload["matched_contrasts"]
    ]
    assert min(measured_ordinals) >= (6 + 4) * 16
    # The admitted evidence is unchanged in size: 4 ordinals x 2 reps x 8
    # steps x 4 contrasts.
    assert len(payload["matched_contrasts"]) == 256


def test_balanced_gate_skips_burn_in_without_a_warmup_block(monkeypatch):
    fake = _SwitchingRealizationPair(dirty_pairs=0)
    monkeypatch.setattr(balanced_gate, "run_schedule_pair", fake)
    reference, runner = _endpoints()
    payload = run_balanced_logits_gate(
        reference,
        runner,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=HORIZON_8,
        repetitions=2,
        warmup_pair_traces=0,
        burn_in_pair_traces=4,
    )

    burn_in = payload["burn_in"]
    assert burn_in["executed"] is False
    assert burn_in["executed_pair_traces"] == 0
    assert burn_in["pair_traces_requested"] == 4
    assert burn_in["pair_results"] == []


def test_noise_floor_burn_in_protects_the_last_two_assertion(monkeypatch):
    monkeypatch.setattr(
        balanced_gate, "run_schedule_pair", _SwitchingRealizationPair(dirty_pairs=2)
    )
    reference, runner = _endpoints()
    with pytest.raises(RuntimeError, match="noise-floor last two traces are unstable"):
        run_alternating_noise_floor(
            reference,
            runner,
            prompt_token_ids=(1, 2, 3, 4),
            forced_token_ids=HORIZON_8,
            repetitions=3,
            warmup_pair_traces=6,
            burn_in_pair_traces=0,
        )

    monkeypatch.setattr(
        balanced_gate, "run_schedule_pair", _SwitchingRealizationPair(dirty_pairs=2)
    )
    payload = run_alternating_noise_floor(
        reference,
        runner,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=HORIZON_8,
        repetitions=3,
        warmup_pair_traces=6,
        burn_in_pair_traces=4,
    )
    assert payload["last_two_pair_traces_exact"] is True
    assert payload["protocol"] == "SPEC-02-alternating-noise-floor-h8-v3"
    assert payload["burn_in"]["executed_pair_traces"] == 4


class _OscillatingPair:
    """Fake ``run_schedule_pair`` reproducing the a40-2026-08-27-r1 pattern.

    The selected (left, right) endpoint pair alternates between two
    realisations on every captured pair trace — a sustained period-2
    oscillation, not a transient — while every other pair is stable.
    """

    def __init__(self, oscillating: tuple[str, str]) -> None:
        self.oscillating = oscillating
        self.captured_by_pair: dict[tuple[str, str], int] = {}

    def __call__(
        self,
        calendar,
        left,
        right,
        *,
        prompt_token_ids,
        forced_token_ids,
        capture,
        cpu_logits_observer=None,
        **_kwargs,
    ):
        if not capture:
            return None
        key = (left.name, right.name)
        count = self.captured_by_pair.get(key, 0)
        self.captured_by_pair[key] = count + 1
        realization = count % 2 if key == self.oscillating else 0
        steps = []
        for step in range(len(forced_token_ids)):
            records = {}
            for within_step, side in enumerate(("left", "right")):
                endpoint = left if side == "left" else right
                metadata = {
                    "calendar": calendar,
                    "endpoint": endpoint.name,
                    "side": side,
                    "decode_step": step,
                    "step": step,
                    "pair_local_forward_ordinal": 2 * step + within_step,
                    "within_step_ordinal": within_step,
                }
                logits = torch.tensor([float(realization), 1.0 + float(realization)])
                if cpu_logits_observer is not None:
                    cpu_logits_observer(dict(metadata), logits)
                records[side] = {
                    **metadata,
                    "sha256": f"{key[0]}-{key[1]}-r{realization}",
                    "top1": 1,
                }
            steps.append(
                {
                    "step": step,
                    "point": "logits",
                    "left": records["left"],
                    "right": records["right"],
                    "comparison": {
                        "max_abs_delta": 0.0,
                        "kl_next_token": 0.0,
                        "left_top1": 1,
                        "right_top1": 1,
                        "top1_agreement": True,
                        "exact": True,
                        "first_coordinate": None,
                        "left_value": None,
                        "right_value": None,
                    },
                }
            )
        return {
            "left_path_fingerprint": f"{key[0]}-{key[1]}-left-r{realization}",
            "right_path_fingerprint": f"{key[0]}-{key[1]}-right-r{realization}",
            "steps": steps,
            "cache_independence": {
                "cache_objects_distinct": True,
                "cache_storage_disjoint": True,
            },
            "autograd_disabled_all_forwards": True,
        }


def test_noise_floor_mixed_pair_oscillation_is_diagnostic_only(monkeypatch):
    """The measured RN period-2 oscillation must not block the floor pairs."""
    monkeypatch.setattr(
        balanced_gate, "run_schedule_pair", _OscillatingPair(("reference", "runner"))
    )
    reference, runner = _endpoints()
    payload = run_alternating_noise_floor(
        reference,
        runner,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=HORIZON_8,
        repetitions=3,
        warmup_pair_traces=6,
        burn_in_pair_traces=4,
    )

    assert payload["protocol"] == "SPEC-02-alternating-noise-floor-h8-v3"
    assert payload["blocking_pairs"] == ["reference_reference", "runner_runner"]
    assert payload["mixed_pair_stability_is_diagnostic_only"] is True
    assert payload["last_two_pair_traces_exact"] is True
    assert payload["pair_stability"]["reference_reference"] is True
    assert payload["pair_stability"]["runner_runner"] is True
    # The oscillation stays visible in the recorded evidence.
    assert payload["pair_stability"]["reference_runner"] is False
    # The floor itself only ever contains the blocking pairs.
    assert {item["pair"] for item in payload["raw_control_floor"]} == {
        "reference_reference",
        "runner_runner",
    }


def test_noise_floor_still_blocks_on_an_unstable_floor_pair(monkeypatch):
    monkeypatch.setattr(
        balanced_gate, "run_schedule_pair", _OscillatingPair(("runner", "runner"))
    )
    reference, runner = _endpoints()
    with pytest.raises(RuntimeError, match="noise-floor last two traces are unstable"):
        run_alternating_noise_floor(
            reference,
            runner,
            prompt_token_ids=(1, 2, 3, 4),
            forced_token_ids=HORIZON_8,
            repetitions=3,
            warmup_pair_traces=6,
            burn_in_pair_traces=4,
        )


def _toy_session(warmups_per_shape: int = 1) -> _MeasurementSession:
    model = toy_model(seed=45)
    handle = SimpleNamespace(
        model=model,
        view=HybridGroupView.from_text_config(model.config),
        device=torch.device("cpu"),
    )
    config = SimpleNamespace(
        numerics=SimpleNamespace(warmup_traces_per_shape=warmups_per_shape),
        identity=SimpleNamespace(
            decode_tokens=3,
            burn_in_repetitions=1,
            measurement_repetitions=3,
        ),
    )
    return _MeasurementSession(handle, config)


def test_warm_uses_one_ledger_per_endpoint():
    session = _toy_session()
    prompt = SimpleNamespace(id="p", token_ids=(1, 2, 3, 4), length_class="short")
    path = CampaignPath(prompt, ExecutionMode.PREFILL_FULL)

    first = session.warm(path, (5, 6, 7), 3)
    # Identical shapes on both sides previously left the runner unwarmed
    # (shared ledger); each endpoint now warms its own path.
    assert first == {"reference": 1, "runner": 1, "total": 2}
    assert session.warm(path, (5, 6, 7), 3) == {
        "reference": 0,
        "runner": 0,
        "total": 0,
    }


def test_measure_forced_burn_in_runs_once_after_a_warmup_and_is_excluded():
    session = _toy_session()
    prompt = SimpleNamespace(id="p", token_ids=(1, 2, 3, 4), length_class="short")
    path = CampaignPath(prompt, ExecutionMode.DECODE_CACHED)

    payload = session.measure_forced(
        case_id="case",
        phase="short",
        path=path,
        forced_token_ids=(5, 6, 7),
        repetitions=2,
        sampling=SamplingMode.GREEDY,
        continuation_seed=None,
        exact_required=False,
        endpoints=None,
        logits_only=False,
        decode_steps=3,
    )

    assert payload["warmup_paths"] == 2
    assert payload["warmup_paths_by_endpoint"] == {"reference": 1, "runner": 1}
    burn_in = payload["burn_in"]
    assert burn_in["executed"] is True
    assert burn_in["executed_repetitions"] == 1
    assert burn_in["excluded_from_blocking_criteria"] is True
    assert all(item["burn_in"] is True for item in burn_in["observations"])
    assert len(payload["repetitions"]) == 2
    assert all("burn_in" not in item for item in payload["repetitions"])

    # Same shapes again: no warmup block, therefore no burn-in.
    second = session.measure_forced(
        case_id="case2",
        phase="short",
        path=path,
        forced_token_ids=(5, 6, 7),
        repetitions=2,
        sampling=SamplingMode.GREEDY,
        continuation_seed=None,
        exact_required=False,
        endpoints=None,
        logits_only=False,
        decode_steps=3,
    )
    assert second["warmup_paths"] == 0
    assert second["burn_in"]["executed"] is False
    assert second["burn_in"]["observations"] == []
