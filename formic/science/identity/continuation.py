"""Reference-generated continuations forced identically through both paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass(frozen=True)
class ForcedContinuation:
    mode: Literal["greedy", "seeded_sampling"]
    seed: int
    token_ids: tuple[int, ...]


def choose_token(
    logits: torch.Tensor,
    *,
    mode: Literal["greedy", "seeded_sampling"],
    generator: torch.Generator,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
) -> int:
    if logits.ndim != 1:
        raise ValueError("next-token logits must be one-dimensional")
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    if mode == "greedy":
        return int(torch.argmax(values).item())
    if mode != "seeded_sampling":
        raise ValueError(f"unknown continuation mode {mode!r}")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("invalid sampling parameters")
    values = values / temperature
    if top_k > 0 and top_k < values.numel():
        cutoff = torch.topk(values, top_k).values[-1]
        values = values.masked_fill(values < cutoff, float("-inf"))
    sorted_logits, sorted_indices = torch.sort(values, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    selected = int(torch.multinomial(sorted_probabilities, 1, generator=generator).item())
    return int(sorted_indices[selected].item())


def generate_forced_continuation(
    *,
    prompt_token_ids: tuple[int, ...],
    steps: int,
    seed: int,
    mode: Literal["greedy", "seeded_sampling"],
    reference_next_logits: Callable[[tuple[int, ...]], torch.Tensor],
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
) -> ForcedContinuation:
    """Generate once from the reference; callers persist then force these IDs."""
    if not prompt_token_ids or steps <= 0 or seed < 0:
        raise ValueError("invalid continuation request")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    context = list(prompt_token_ids)
    generated: list[int] = []
    for _ in range(steps):
        logits = reference_next_logits(tuple(context))
        token = choose_token(
            logits,
            mode=mode,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated.append(token)
        context.append(token)
    return ForcedContinuation(mode=mode, seed=seed, token_ids=tuple(generated))
