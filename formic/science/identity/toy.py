"""Tiny stock Qwen instance for weight-free SPEC-02 end-to-end checks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import formic  # noqa: F401 - torch 2.4 / transformers 5.8 compatibility shim

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CONFIG = (
    REPO_ROOT / "configs" / "checkpoint_metadata" / "qwen3_8_27b" / "config.json"
)


def toy_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.loads(OFFICIAL_CONFIG.read_text(encoding="utf-8"))["text_config"]
    raw.update(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        max_position_embeddings=256,
        dtype="float32",
        partial_rotary_factor=0.5,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    )
    return Qwen3_5TextConfig(**raw)


def toy_model(seed: int = 0):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(toy_text_config()).eval()
