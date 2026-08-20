"""Weight-free tests for SPEC-01 same-path and ordinal controls."""

from scripts.step1_ordinal_noise_controls import (
    INSTANCE_NAMES,
    _aggregate_comparisons,
    _ordinal_path_order,
    _rotated_instances,
)


def test_instance_order_rotates_without_dropping_an_instance():
    orders = [_rotated_instances(cycle) for cycle in range(4)]
    assert [order[0] for order in orders] == list(INSTANCE_NAMES)
    assert all(set(order) == set(INSTANCE_NAMES) for order in orders)


def test_ordinal_order_alternates_from_requested_phase():
    assert _ordinal_path_order(0, "runner-first") == ("formic_runner", "hf_explicit")
    assert _ordinal_path_order(1, "runner-first") == ("hf_explicit", "formic_runner")
    assert _ordinal_path_order(0, "explicit-first") == ("hf_explicit", "formic_runner")
    assert _ordinal_path_order(1, "explicit-first") == ("formic_runner", "hf_explicit")


def test_aggregate_comparisons_sums_prompt_metrics():
    prompts = {
        "a": {
            "comparisons": {
                "x": {"logits": {"exact_steps": 2, "top1_matches": 3, "steps": 4}}
            }
        },
        "b": {
            "comparisons": {
                "x": {"logits": {"exact_steps": 5, "top1_matches": 6, "steps": 8}}
            }
        },
    }
    assert _aggregate_comparisons(prompts, "x") == {
        "exact_steps": 7,
        "top1_matches": 9,
        "steps": 12,
    }
