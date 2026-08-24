from __future__ import annotations

import gc
import json
import weakref

import torch
import pytest

from formic.backbone.groups import HybridGroupView
from formic.science.identity.executor import Endpoint
from formic.science.identity import schedule_diagnostic
from formic.science.identity.artifacts import ArtifactError
from formic.science.identity.crossover_diagnostic import (
    AttemptMemoryWriter,
    CPULogitBank,
    CrossoverBlocked,
    CrossoverIdentity,
    CrossoverWriter,
    balanced_design,
    assert_resumable_terminal,
    build_analysis,
    build_matched_slot_evidence,
    build_ordinal_position_observations,
    compare_matched_logits,
    prepare_attempt_metadata,
    process_lifetime_diagnostic_forward_ordinal,
    round_configurations,
    round_relative_global_forward_ordinal,
    validate_balanced_design,
)
from tests.toy_qwen import toy_model


def _endpoints():
    model = toy_model(seed=47)
    view = HybridGroupView.from_text_config(model.config)
    return (
        Endpoint("reference", model, view, False),
        Endpoint("runner", model, view, True),
    )


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_alternating_warmup_disables_autograd_releases_outputs_and_records_order(monkeypatch):
    left, right = _endpoints()
    original = schedule_diagnostic._call_endpoint
    output_refs = []
    grad_observations = []

    def observed_call(endpoint, input_ids, cache):
        grad_observations.append((endpoint.name, torch.is_grad_enabled()))
        output = original(endpoint, input_ids, cache)
        output_refs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(schedule_diagnostic, "_call_endpoint", observed_call)
    events = []
    memory_labels = []
    result = schedule_diagnostic.run_schedule_pair(
        "alternating",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=False,
        event_observer=events.append,
        memory_observer=memory_labels.append,
    )
    gc.collect()

    assert result is None
    assert grad_observations == [
        ("reference", False), ("runner", False),
        ("reference", False), ("runner", False),
        ("reference", False), ("runner", False),
    ]
    assert all(reference() is None for reference in output_refs)
    assert [
        (event["side"], event["step"])
        for event in events
        if event["event"] == "after_endpoint"
    ] == [
        ("left", 0), ("right", 0),
        ("left", 1), ("right", 1),
        ("left", 2), ("right", 2),
    ]
    left_cache_ids = {
        event["cache_object_id"] for event in events if event["side"] == "left"
    }
    right_cache_ids = {
        event["cache_object_id"] for event in events if event["side"] == "right"
    }
    assert len(left_cache_ids) == len(right_cache_ids) == 1
    assert left_cache_ids.isdisjoint(right_cache_ids)
    assert memory_labels == [
        "before_cache_creation",
        "after_left_step_0", "after_left_step_0_output_deleted",
        "after_right_step_0", "after_right_step_0_output_deleted",
        "after_left_step_1", "after_left_step_1_output_deleted",
        "after_right_step_1", "after_right_step_1_output_deleted",
        "after_left_step_2", "after_left_step_2_output_deleted",
        "after_right_step_2", "after_right_step_2_output_deleted",
        "after_warmup",
    ]


def test_measured_alternating_result_has_independent_caches_and_no_tensors():
    left, right = _endpoints()
    result = schedule_diagnostic.run_schedule_pair(
        "alternating",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=True,
    )

    assert result is not None
    assert result["autograd_disabled_all_forwards"] is True
    assert result["cache_independence"] == {
        "cache_objects_distinct": True,
        "cache_storage_disjoint": True,
        "fresh_cache_pair_constructed_for_call": True,
    }
    assert not _contains_tensor(result)
    assert all(step["left"]["device"] == "cpu" for step in result["steps"])
    assert all(step["right"]["device"] == "cpu" for step in result["steps"])
    json.dumps(result)


def test_sequential_calendar_runs_whole_left_path_before_right_path():
    left, right = _endpoints()
    result = schedule_diagnostic.run_schedule_pair(
        "sequential",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7),
        capture=True,
    )

    assert result is not None
    assert [(item["side"], item["step"]) for item in result["forward_order"]] == [
        ("left", 0),
        ("left", 1),
        ("left", 2),
        ("right", 0),
        ("right", 1),
        ("right", 2),
    ]


@pytest.mark.parametrize(
    ("calendar", "first_sides"),
    (
        ("abba", ("left", "right", "right", "left") * 2),
        ("baab", ("right", "left", "left", "right") * 2),
    ),
)
def test_balanced_calendars_execute_both_sides_with_exact_ordinal_metadata(
    calendar, first_sides
):
    left, right = _endpoints()
    observed = []
    result = schedule_diagnostic.run_schedule_pair(
        calendar,
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6, 7, 8, 9, 10, 11, 12),
        capture=True,
        event_observer=observed.append,
    )

    assert result is not None
    order = result["forward_order"]
    assert [item["pair_local_forward_ordinal"] for item in order] == list(range(16))
    assert [item["side"] for item in order[::2]] == list(first_sides)
    assert [item["within_step_ordinal"] for item in order] == [0, 1] * 8
    for ordinal, item in enumerate(order):
        assert item == {
            "calendar": calendar,
            "endpoint": "reference" if item["side"] == "left" else "runner",
            "side": item["side"],
            "decode_step": ordinal // 2,
            "step": ordinal // 2,
            "pair_local_forward_ordinal": ordinal,
            "within_step_ordinal": ordinal % 2,
        }
    logit_records = [
        step[side]
        for step in result["steps"]
        for side in ("left", "right")
    ]
    assert {item["pair_local_forward_ordinal"] for item in logit_records} == set(range(16))
    assert all(item["calendar"] == calendar for item in logit_records)
    assert not _contains_tensor(result)
    assert all(not event["grad_enabled"] for event in observed)


def test_cpu_logits_observer_is_detached_cpu_and_result_remains_tensor_free():
    left, right = _endpoints()
    observations = []

    def observe(metadata, logits):
        observations.append((metadata, logits.device.type, logits.requires_grad, logits.clone()))

    result = schedule_diagnostic.run_schedule_pair(
        "abba",
        left,
        right,
        prompt_token_ids=(1, 2, 3, 4),
        forced_token_ids=(5, 6),
        capture=True,
        cpu_logits_observer=observe,
    )

    assert result is not None and not _contains_tensor(result)
    assert len(observations) == 4
    assert all(device == "cpu" and requires_grad is False for _, device, requires_grad, _ in observations)
    assert [item[0]["pair_local_forward_ordinal"] for item in observations] == list(range(4))


def test_balanced_design_has_four_pairs_and_complete_latin_slot_coverage():
    design = balanced_design()
    validation = validate_balanced_design(design)

    assert len(design) == 64
    assert validation["valid"] is True
    assert {item.pair for item in design} == {
        "reference_reference",
        "runner_runner",
        "reference_runner",
        "runner_reference",
    }
    assert [item.calendar for item in round_configurations(0)] == ["abba"] * 4 + ["baab"] * 4
    assert [item.calendar for item in round_configurations(4)] == ["baab"] * 4 + ["abba"] * 4
    assert all(ordinals == list(range(8)) for ordinals in validation["coverage"].values())


def test_same_slot_matcher_rejects_ordinal_mismatch():
    metadata = {
        "calendar": "abba",
        "round": 0,
        "configuration_ordinal": 2,
        "decode_step": 3,
        "side": "left",
        "pair_local_forward_ordinal": 6,
        "within_step_ordinal": 0,
        "round_relative_global_forward_ordinal": 102,
        "process_lifetime_diagnostic_forward_ordinal": 198,
        "repetition": 0,
    }
    candidate = {**metadata, "round_relative_global_forward_ordinal": 103}

    with pytest.raises(CrossoverBlocked, match="ordinal"):
        compare_matched_logits(metadata, torch.tensor([1.0]), candidate, torch.tensor([1.0]))
    wrong_process = {
        **metadata,
        "process_lifetime_diagnostic_forward_ordinal": 199,
    }
    with pytest.raises(CrossoverBlocked, match="process-lifetime"):
        compare_matched_logits(
            metadata,
            torch.tensor([1.0]),
            wrong_process,
            torch.tensor([1.0]),
        )


def test_round_relative_global_ordinal_includes_repetition_and_spans_one_round():
    assert round_relative_global_forward_ordinal(0, 0, 0) == 0
    assert round_relative_global_forward_ordinal(0, 1, 0) == 16
    assert round_relative_global_forward_ordinal(1, 0, 0) == 48
    assert round_relative_global_forward_ordinal(7, 2, 15) == 383


def test_process_lifetime_diagnostic_ordinal_boundaries_include_warmups():
    assert process_lifetime_diagnostic_forward_ordinal(0, 0, 0, 0) == 96
    assert process_lifetime_diagnostic_forward_ordinal(0, 7, 2, 15) == 479
    assert process_lifetime_diagnostic_forward_ordinal(1, 0, 0, 0) == 480
    assert process_lifetime_diagnostic_forward_ordinal(7, 7, 2, 15) == 3167


def test_matched_slot_evidence_records_distinct_process_positions_across_rounds():
    bank = CPULogitBank()

    def add(round_index, pair, endpoint):
        metadata = {
            "calendar": "abba",
            "round": round_index,
            "configuration_ordinal": 0,
            "pair": pair,
            "repetition": 0,
            "endpoint": endpoint,
            "side": "left",
            "decode_step": 0,
            "within_step_ordinal": 0,
            "pair_local_forward_ordinal": 0,
            "round_relative_global_forward_ordinal": 0,
            "process_lifetime_diagnostic_forward_ordinal": (
                process_lifetime_diagnostic_forward_ordinal(round_index, 0, 0, 0)
            ),
        }
        bank.add(metadata, torch.tensor([1.0, 2.0]))

    add(0, "reference_reference", "reference")
    add(3, "runner_reference", "runner")
    reference = bank.get("abba", 0, "reference_reference", 0, "left", 0)
    candidate = bank.get("abba", 0, "runner_reference", 0, "left", 0)
    evidence = build_matched_slot_evidence(*reference, *candidate)

    assert evidence["round_relative_calendar_slot_matched"] is True
    assert evidence["matching_scope"] == "round_relative_balanced_crossover"
    assert evidence["round_relative_global_forward_ordinal"] == 0
    assert (evidence["reference_round"], evidence["candidate_round"]) == (0, 3)
    assert evidence["reference_process_lifetime_diagnostic_forward_ordinal"] == 96
    assert evidence["candidate_process_lifetime_diagnostic_forward_ordinal"] == 1248
    assert evidence["process_lifetime_diagnostic_forward_ordinals_matched"] is False


def _crossover_identity(**changes):
    values = {
        "protocol": "test",
        "config_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "corpus_source_sha256": "c" * 64,
        "git_commit": "d" * 40,
        "backbone_sha256": "e" * 64,
    }
    values.update(changes)
    return CrossoverIdentity(**values)


def test_crossover_writer_resume_identity_and_corruption(tmp_path):
    root = tmp_path / "run"
    writer = CrossoverWriter(root, _crossover_identity())
    payload = {"schema_version": 1, "round": 0, "values": [1, 2]}
    writer.write_configuration("config_0", payload)
    writer.write_round(0, payload)
    writer.validate()
    with pytest.raises(ArtifactError, match="cannot contain tensors"):
        writer.write_diagnostic("tensor", {"value": torch.tensor([1.0])})

    resumed = CrossoverWriter(root, _crossover_identity())
    resumed.write_configuration("config_0", payload)
    resumed.write_round(0, payload)
    assert resumed.completed_rounds() == frozenset({0})
    with pytest.raises(ArtifactError, match="identity differs"):
        CrossoverWriter(root, _crossover_identity(config_sha256="f" * 64))

    (root / "rounds" / "0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checkpoint invalid"):
        resumed.validate()


def _analysis_configurations():
    configurations = []
    for item in balanced_design():
        repetitions = []
        for repetition in range(3):
            slots = []
            for step in range(8):
                for side in ("left", "right"):
                    slots.append(
                        {
                            "calendar": item.calendar,
                            "pair": item.pair,
                            "repetition": repetition,
                            "side": side,
                            "decode_step": step,
                            "sha256": (
                                f"{item.calendar}:{item.pair}:{repetition}:{side}:{step}"
                            ),
                            "pair_comparison": {
                                "exact": True,
                                "max_abs_delta": 0.0,
                                "kl_next_token": 0.0,
                                "first_coordinate": None,
                            },
                        }
                    )
            repetitions.append({"repetition": repetition, "slots": slots})
        configurations.append(
            {
            **item.to_dict(),
            "status": "COMPLETE",
            "all_stability_exact": True,
            "all_last_two_slots_exact": True,
            "repetitions": repetitions,
            }
        )
    return configurations


def test_analysis_readiness_requires_every_exact_control_and_slot():
    exact = {
        "metric": {
            "exact": True,
            "max_abs_delta": 0.0,
            "kl_next_token": 0.0,
            "first_coordinate": None,
        }
    }
    ready = build_analysis(
        configurations=_analysis_configurations(),
        same_slot_contrasts=[exact] * 1536,
        inversion_checks=[exact] * 768,
        design_validation=validate_balanced_design(balanced_design()),
    )

    assert ready["status"] == "COMPLETE"
    assert ready["readiness"]["status"] == "READY"
    assert ready["readiness"]["ready_for_full_spec_02_campaign"] is True
    assert ready["readiness"]["official_command_recommendation"] is None
    assert ready["readiness"]["official_launcher_requires_calendar_adaptation"] is True
    assert ready["readiness"]["currently_runnable"] is False
    assert ready["readiness"]["command_template_after_code_adaptation"].startswith(
        "python scripts/step2_a40_campaign.py"
    )
    assert ready["counts"]["measured_pair_traces"] == 192
    assert ready["same_slot_metric_summary"]["expected_count"] == 1536
    assert ready["inversion_metric_summary"]["expected_count"] == 768
    assert len(ready["pair_result_summaries"]) == 4
    assert len(ready["questions_fr"]) == 6
    assert [item["question_id"][0] for item in ready["questions_fr"]] == list("abcdef")

    configurations = _analysis_configurations()
    configurations[0]["all_last_two_slots_exact"] = False
    blocked = build_analysis(
        configurations=configurations,
        same_slot_contrasts=[exact] * 1536,
        inversion_checks=[exact] * 768,
        design_validation=validate_balanced_design(balanced_design()),
    )
    assert blocked["status"] == "COMPLETE"
    assert blocked["readiness"]["status"] == "BLOCKED"
    assert blocked["readiness"]["official_command_recommendation"] is None
    assert blocked["readiness"]["command_template_after_code_adaptation"] is None

    nonexact = {
        "calendar": "abba",
        "configuration_ordinal": 0,
        "repetition": 0,
        "step": 0,
        "side": "left",
        "metric": {
            "exact": False,
            "max_abs_delta": 0.25,
            "kl_next_token": 0.5,
            "first_coordinate": [7],
            "reference_value": 1.0,
            "candidate_value": 1.25,
        },
    }
    numerical_block = build_analysis(
        configurations=_analysis_configurations(),
        same_slot_contrasts=[nonexact] + [exact] * 1535,
        inversion_checks=[exact] * 768,
        design_validation=validate_balanced_design(balanced_design()),
    )
    assert numerical_block["status"] == "COMPLETE"
    assert numerical_block["readiness"]["status"] == "BLOCKED"
    assert numerical_block["same_slot_metric_summary"]["nonexact_count"] == 1
    assert numerical_block["same_slot_metric_summary"]["max_max_abs_delta"] == 0.25
    assert numerical_block["same_slot_metric_summary"]["max_kl_next_token"] == 0.5
    assert numerical_block["same_slot_metric_summary"]["first_nonexact"][
        "first_coordinate"
    ] == [7]


def test_ordinal_position_observations_report_hash_association_without_cause():
    configurations = _analysis_configurations()
    changed = next(
        item
        for item in configurations
        if item["calendar"] == "abba"
        and item["pair"] == "reference_reference"
        and item["configuration_ordinal"] == 1
    )
    changed["repetitions"][0]["slots"][0]["sha256"] = "changed"

    observations = build_ordinal_position_observations(configurations)
    target = next(
        item
        for item in observations["observations"]
        if item["calendar"] == "abba"
        and item["pair"] == "reference_reference"
        and item["repetition"] == 0
        and item["side"] == "left"
        and item["decode_step"] == 0
    )
    assert observations["aggregate"]["expected_groups"] == 384
    assert observations["aggregate"]["exact_coverage_groups"] == 384
    assert observations["aggregate"]["any_observed_hash_change_by_ordinal"] is True
    assert target["first_differing_ordinal_hash"]["differing_ordinal"] == 1
    assert observations["causal_attribution"] is None


def test_attempt_metadata_and_memory_initialization_preserve_prior_evidence(tmp_path):
    metadata_path = tmp_path / "run_metadata.json"
    base = {"schema_version": 1, "kind": "test", "protocol": {"repetitions": 3}}
    first, first_id = prepare_attempt_metadata(metadata_path, base, {"resume": False})
    second, second_id = prepare_attempt_metadata(metadata_path, base, {"resume": True})
    assert first_id == "attempt_000"
    assert second_id == "attempt_001"
    assert first["attempts"] == [{"attempt_id": "attempt_000", "resume": False}]
    assert second["attempts"] == [
        {"attempt_id": "attempt_000", "resume": False},
        {"attempt_id": "attempt_001", "resume": True},
    ]

    memory_path = tmp_path / "cuda_memory.json"
    memory_path.write_text(
        json.dumps({"schema_version": 1, "measurements": [{"label": "attempt_000:before"}]}),
        encoding="utf-8",
    )
    memory = AttemptMemoryWriter(memory_path, "attempt_001")
    assert memory.measurements == [{"label": "attempt_000:before"}]


def test_resume_terminal_refuses_complete_and_allows_failed_or_partial(tmp_path):
    terminal = tmp_path / "terminal.json"
    assert_resumable_terminal(terminal)
    terminal.write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
    assert_resumable_terminal(terminal)
    terminal.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    assert_resumable_terminal(terminal)
    terminal.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="already COMPLETE"):
        assert_resumable_terminal(terminal)


def test_incomplete_analysis_answers_scientific_questions_as_unknown():
    analysis = build_analysis(
        configurations=[],
        same_slot_contrasts=[],
        inversion_checks=[],
        design_validation=validate_balanced_design(balanced_design()),
    )
    answers = {item["question_id"][0]: item["answer"] for item in analysis["questions_fr"]}
    assert analysis["status"] == "BLOCKED"
    assert answers["c"] is None
    assert answers["d"] is None
    assert answers["e"] is None
    assert answers["f"] is None
