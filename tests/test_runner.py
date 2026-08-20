"""SPEC-01 input boundary: text-only, batch 1, no padding."""

from __future__ import annotations

import pytest
import torch

from formic.backbone.runner import _validate_batch1_no_padding, forced_cached_decode_logits


def test_batch_one_without_padding_is_accepted():
    ids = torch.tensor([[1, 2, 3]])
    _validate_batch1_no_padding(ids)
    _validate_batch1_no_padding(ids, torch.ones_like(ids))


def test_batching_is_rejected():
    with pytest.raises(ValueError, match="batch"):
        _validate_batch1_no_padding(torch.tensor([[1, 2], [3, 4]]))


def test_padding_is_rejected():
    ids = torch.tensor([[1, 2, 0]])
    with pytest.raises(ValueError, match="padded"):
        _validate_batch1_no_padding(ids, torch.tensor([[1, 1, 0]]))


def test_mismatched_attention_mask_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        _validate_batch1_no_padding(torch.tensor([[1, 2]]), torch.tensor([[1]]))


def test_forced_cached_decode_rejects_an_empty_continuation():
    with pytest.raises(ValueError, match="must not be empty"):
        forced_cached_decode_logits(object(), (1, 2), ())
