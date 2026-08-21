from __future__ import annotations

import torch
from types import SimpleNamespace

from formic.backbone.runner import forced_cached_decode_logits
from formic.science.identity.continuation import (
    choose_token,
    generate_forced_continuation,
)
from formic.science.identity.toy import toy_model


def test_greedy_continuation_is_reference_generated_once():
    calls = []

    def logits(context):
        calls.append(context)
        values = torch.zeros(8)
        values[(len(context) + 1) % 8] = 10
        return values

    result = generate_forced_continuation(
        prompt_token_ids=(1, 2),
        steps=3,
        seed=17,
        mode="greedy",
        reference_next_logits=logits,
    )
    assert result.token_ids == (3, 4, 5)
    assert calls == [(1, 2), (1, 2, 3), (1, 2, 3, 4)]


def test_seeded_sampling_is_reproducible():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        return [
            choose_token(
                logits,
                mode="seeded_sampling",
                generator=generator,
                top_k=4,
                top_p=1.0,
            )
            for _ in range(10)
        ]

    assert draw(9) == draw(9)
    assert draw(9) != draw(10)


def test_forced_inference_logits_are_seed_independent_on_stock_toy_model():
    model = toy_model(seed=53)
    handle = SimpleNamespace(model=model)
    prompt = (1, 2, 3, 4, 5, 6, 7, 8)
    continuation = (9, 10, 11, 12)

    torch.manual_seed(101)
    first = forced_cached_decode_logits(handle, prompt, continuation)
    torch.manual_seed(909)
    second = forced_cached_decode_logits(handle, prompt, continuation)

    assert len(first) == len(second) == len(continuation)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
