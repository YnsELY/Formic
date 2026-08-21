"""Shared stock-Qwen toy model and synthetic hybrid-cache fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import formic  # noqa: F401 - torch 2.4 / transformers 5.8 compatibility shim
from formic.backbone import constants as C
from formic.science.identity.toy import toy_model, toy_text_config

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG = (
    REPO_ROOT / "configs" / "checkpoint_metadata" / "qwen3_8_27b" / "config.json"
)


def audited_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))["text_config"]
    return Qwen3_5TextConfig(**raw)


class CacheModelStub(nn.Module):
    """Minimal model owner used to test cache state without Qwen compute."""

    def __init__(self, config: Any, *, rope_deltas: torch.Tensor | None | object = ...):
        super().__init__()
        self.config = config
        if rope_deltas is not ...:
            self.rope_deltas = rope_deltas

    def forward(self, input_ids=None, past_key_values=None, use_cache=True, **kwargs):
        # Model the audited in-place recurrent write that makes A4 necessary.
        past_key_values.layers[0].recurrent_states.add_(1)
        return SimpleNamespace(
            logits=torch.zeros(1, input_ids.shape[-1], 8),
            past_key_values=past_key_values,
        )


def synthetic_cache(config: Any, *, sequence_length: int = 3, dtype=torch.bfloat16):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=config)
    for index in range(C.NUM_LAYERS):
        value = float(index + 1)
        if index in C.ATTENTION_LAYER_INDICES:
            keys = torch.full(
                (1, config.num_key_value_heads, sequence_length, config.head_dim),
                value,
                dtype=dtype,
            )
            values = torch.full_like(keys, value + 0.5)
            cache.update(keys, values, index)
        else:
            mixed_width = (
                2 * config.linear_num_key_heads * config.linear_key_head_dim
                + config.linear_num_value_heads * config.linear_value_head_dim
            )
            conv = torch.full(
                (1, mixed_width, config.linear_conv_kernel_dim), value, dtype=dtype
            )
            recurrent = torch.full(
                (
                    1,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
                value + 0.25,
                dtype=dtype,
            )
            cache.update_conv_state(conv, index)
            cache.update_recurrent_state(recurrent, index)
    return cache


def cache_tensors(cache: Any) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(cache.layers):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                tensors[f"layers[{index}].{name}"] = value
    return tensors
