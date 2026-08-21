from __future__ import annotations

import torch

from formic.science.identity.metrics import compare_logits, compare_tensors


def test_tensor_comparison_reports_exact_and_first_coordinate():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    exact = compare_tensors(reference, reference.clone())
    assert exact.exact
    assert exact.max_abs_delta == 0
    assert exact.first_coordinate is None

    candidate = reference.clone()
    candidate[1, 0] = 3.5
    changed = compare_tensors(reference, candidate)
    assert not changed.exact
    assert changed.max_abs_delta == 0.5
    assert changed.first_coordinate == (1, 0)
    assert changed.reference_value == 3.0
    assert changed.candidate_value == 3.5


def test_dtype_difference_is_not_exact_even_when_values_match():
    result = compare_tensors(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    assert not result.exact
    assert result.max_abs_delta == 0


def test_logit_comparison_reports_kl_and_top1():
    reference = torch.tensor([2.0, 1.0, 0.0])
    same = compare_logits(reference, reference.clone())
    assert same.tensor.exact
    assert same.kl_next_token == 0
    assert same.top1_agreement

    changed = compare_logits(reference, torch.tensor([1.0, 3.0, 0.0]))
    assert changed.kl_next_token > 0
    assert not changed.top1_agreement
    assert changed.reference_top1 == 0
    assert changed.candidate_top1 == 1


def test_shape_and_rank_mismatches_fail_loudly():
    try:
        compare_tensors(torch.zeros(2), torch.zeros(3))
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch was accepted")

    try:
        compare_logits(torch.zeros(1, 2), torch.zeros(1, 2))
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("rank mismatch was accepted")
