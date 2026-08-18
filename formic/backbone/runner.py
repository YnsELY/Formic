"""Native generation through Formic, with pinned thinking and sampling policies.

Step 1 only exercises the *stock* generation path: no typed actions, no control
tokens, no scratch/action separation (those arrive in steps 5-6). What this
module guarantees today is that every generation records the policy that
produced it, per plan 2.1/2.2:

* thinking mode (``on`` / ``off`` / ``capped-N``) is pinned and reported;
* control-field sampling is greedy (there are no control fields yet, so this is
  enforced later; the policy object already carries it);
* payload sampling uses the checkpoint defaults until the step-3 sweep decides.

A run is reproducible from (config hash, seed, prompt hash, git commit).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from formic.backbone.loader import BackboneHandle
from formic.config.schema import RunConfig

__all__ = [
    "GenerationResult",
    "ForwardResult",
    "render_chat_prompt",
    "generate",
    "forward_logits",
    "set_seed",
]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can influence a run."""
    from formic.science.determinism import configure_determinism

    configure_determinism(seed, deterministic)


@dataclass
class GenerationResult:
    prompt: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    text: str
    seed: int
    do_sample: bool
    thinking_mode: str
    thinking_cap: int
    sampling: dict[str, Any]
    seconds: float
    config_hash: str
    prompt_hash: str = field(init=False)
    output_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.prompt_hash = _hash_ids(self.prompt_token_ids)
        self.output_hash = _hash_ids(self.generated_token_ids)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prompt_token_ids"] = list(self.prompt_token_ids)
        data["generated_token_ids"] = list(self.generated_token_ids)
        return data


@dataclass
class ForwardResult:
    """A single forward pass, summarised for identity comparisons."""

    input_token_ids: tuple[int, ...]
    last_logits: torch.Tensor
    argmax_id: int
    top_k_ids: tuple[int, ...]
    top_k_values: tuple[float, ...]
    logits_sha256: str
    logits_rms: float
    logits_min: float
    logits_max: float
    seconds: float

    def to_dict(self, include_logits: bool = False) -> dict[str, Any]:
        data = {
            "input_token_ids": list(self.input_token_ids),
            "argmax_id": self.argmax_id,
            "top_k_ids": list(self.top_k_ids),
            "top_k_values": list(self.top_k_values),
            "logits_sha256": self.logits_sha256,
            "logits_rms": self.logits_rms,
            "logits_min": self.logits_min,
            "logits_max": self.logits_max,
            "seconds": self.seconds,
        }
        if include_logits:
            data["last_logits"] = self.last_logits.float().tolist()
        return data


def render_chat_prompt(
    handle: BackboneHandle,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages through the checkpoint's own chat template.

    The template's ``enable_thinking`` switch is driven by the pinned thinking
    policy (plan 2.1). This is also the foundation of TNPR (plan 2.3): every
    Formic packet must be a valid instance of this native template.
    """
    thinking = handle.config.thinking
    return handle.tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=thinking.enable_thinking,
    )


@torch.no_grad()
def generate(
    handle: BackboneHandle,
    prompt: str,
    *,
    do_sample: bool | None = None,
    max_new_tokens: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Generate with the run's pinned policies.

    ``do_sample=False`` gives the greedy path used by identity checks;
    ``do_sample=True`` uses the payload sampling policy of the config.
    """
    config: RunConfig = handle.config
    seed = config.run.seed if seed is None else seed
    set_seed(seed, config.run.deterministic)

    sampling = config.sampling.payload
    do_sample = sampling.do_sample if do_sample is None else do_sample
    max_new_tokens = config.generation.max_new_tokens if max_new_tokens is None else max_new_tokens
    if config.thinking.mode == "capped":
        max_new_tokens = min(max_new_tokens, config.thinking.cap_tokens)

    device = _input_device(handle)
    encoded = handle.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    _validate_batch1_no_padding(input_ids, attention_mask)

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": list(config.generation.eos_token_ids),
        "pad_token_id": handle.tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        kwargs.update(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
        )

    started = time.time()
    output = handle.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    seconds = time.time() - started

    prompt_len = input_ids.shape[-1]
    generated = output[0, prompt_len:].tolist()
    return GenerationResult(
        prompt=prompt,
        prompt_token_ids=tuple(input_ids[0].tolist()),
        generated_token_ids=tuple(generated),
        text=handle.tokenizer.decode(generated, skip_special_tokens=False),
        seed=seed,
        do_sample=do_sample,
        thinking_mode=config.thinking.mode,
        thinking_cap=config.thinking.cap_tokens,
        sampling=asdict(sampling) if do_sample else {"mode": "greedy"},
        seconds=seconds,
        config_hash=config.config_hash(),
    )


@torch.no_grad()
def forward_logits(
    handle: BackboneHandle,
    token_ids: Sequence[int],
    *,
    top_k: int = 10,
) -> ForwardResult:
    """Single forward pass returning last-position logits and their fingerprint.

    The SHA-256 is taken over the FP32 bytes of the logits vector, matching the
    audit's baseline-identity protocol so Formic numbers are comparable with
    ``results/baseline_identity.json``.

    A1 note: ``use_cache=False`` is passed here only to avoid *allocating* a
    cache, and no cache object is supplied. It must never be read as "this pass
    is read-only" — with a cache provided, the GDN and attention layers mutate it
    regardless of the flag. Any future code that hands a cache to the model has
    to snapshot/fork it explicitly (A3, A4).
    """
    device = _input_device(handle)
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    _validate_batch1_no_padding(input_ids)

    started = time.time()
    outputs = handle.model(input_ids=input_ids, use_cache=False)
    seconds = time.time() - started

    logits = outputs.logits[0, -1].detach().to("cpu")
    logits_f32 = logits.float()
    values, indices = torch.topk(logits_f32, k=top_k)
    return ForwardResult(
        input_token_ids=tuple(token_ids),
        last_logits=logits,
        argmax_id=int(torch.argmax(logits_f32).item()),
        top_k_ids=tuple(int(i) for i in indices.tolist()),
        top_k_values=tuple(float(v) for v in values.tolist()),
        logits_sha256=hashlib.sha256(logits_f32.numpy().tobytes()).hexdigest(),
        logits_rms=float(torch.sqrt(torch.mean(logits_f32**2)).item()),
        logits_min=float(logits_f32.min().item()),
        logits_max=float(logits_f32.max().item()),
        seconds=seconds,
    )


@torch.no_grad()
def manual_greedy_decode(
    handle: BackboneHandle,
    token_ids: Sequence[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy decode written out explicitly, bypassing ``model.generate()``.

    SPEC-01 uses this beside native ``generate()`` to make the stock cached
    forward sequence directly observable on both CausalLM executions. Position
    IDs remain ``None`` so the text model follows its documented four-axis
    position contract without wrapper-specific preparation.

    Cache handling: the model creates its own cache (A2 - never build a
    ``DynamicCache`` without the model config), it is used strictly forward, and
    it is never replayed (A1, A3, A4).
    """
    device = _input_device(handle)
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    _validate_batch1_no_padding(input_ids)

    past = None
    generated: list[int] = []
    step_argmax_logits: list[float] = []
    started = time.time()
    current = input_ids
    for _ in range(max_new_tokens):
        outputs = handle.model(input_ids=current, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1].float()
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        step_argmax_logits.append(float(logits[next_id].item()))
        current = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return {
        "input_token_ids": list(token_ids),
        "generated_token_ids": generated,
        "argmax_logit_per_step": step_argmax_logits,
        "text": handle.tokenizer.decode(generated, skip_special_tokens=False),
        "seconds": time.time() - started,
        "cache_seq_length": int(past.get_seq_length()) if past is not None else 0,
    }


def _input_device(handle: BackboneHandle) -> torch.device:
    device = getattr(handle.model, "device", None)
    if isinstance(device, torch.device):
        return device
    return next(handle.model.parameters()).device


def _validate_batch1_no_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> None:
    """Enforce SPEC-01's batch-1, no-padding execution boundary (A8)."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"SPEC-01 requires batch 1 with input_ids shaped [1, sequence], "
            f"got {tuple(input_ids.shape)}"
        )
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids {tuple(input_ids.shape)}"
            )
        if not bool(torch.all(attention_mask == 1)):
            raise ValueError("SPEC-01 forbids padded inputs (A8)")


def _hash_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode("utf-8")).hexdigest()[:16]
