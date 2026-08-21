"""The position-id contract of the text model, and why Formic pins it.

Audit report 05 documents the contract: the text model works with four position
axes — axis 0 for text/causal-mask purposes, axes 1..3 for M-RoPE — and builds
them itself when it receives ``position_ids=None``.

Step 1 found that the two Hugging Face entry points do **not** feed that contract
identically during ``generate()``:

* ``Qwen3_5ForCausalLM`` (Formic's text-only path) produces 2-D position ids,
  which the text model expands to the documented ``[4, B, S]``;
* ``Qwen3_5ForConditionalGeneration`` overrides
  ``_prepare_position_ids_for_generation`` and, in decode, passes ``[1, B, S]``,
  which matches neither branch, so ``text_position_ids`` becomes ``None``.

The multimodal distinction is retained as an upstream regression guard, not as
SPEC-01's active reference path; both current acceptance sides use CausalLM.
These tests are weight-free: only the tiny rotary module and text-model source
are inspected.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import formic  # noqa: F401 - applies the torch/transformers environment shim
from formic.backbone import constants as C

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "checkpoint_metadata"
    / "qwen3_8_27b"
)


@pytest.fixture(scope="module")
def text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))["text_config"]
    return Qwen3_5TextConfig(**raw)


def test_text_model_recognises_the_four_axis_contract(text_config):
    """The `[4, B, S]` contract is what the audited text model actually checks."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    source = inspect.getsource(Qwen3_5TextModel.forward)
    assert "position_ids.ndim == 3 and position_ids.shape[0] == 4" in source, (
        "the text model no longer keys on the 4-axis position contract; audit report 05 "
        "and Formic's decode path both assume it"
    )
    assert "expand(4," in source, "2-D position ids are no longer expanded to 4 axes"


def test_rope_parameters_match_the_audit(text_config):
    params = text_config.rope_parameters
    assert params["rope_theta"] == C.ROPE_THETA
    assert tuple(params["mrope_section"]) == C.MROPE_SECTION
    assert params["mrope_interleaved"] is C.MROPE_INTERLEAVED
    assert params["partial_rotary_factor"] == C.PARTIAL_ROTARY_FACTOR


def test_rotary_is_indifferent_to_axis_count_on_pure_text(text_config):
    """On text, the three M-RoPE axes are identical, so 1 axis == 3 axes.

    This preserves the historical entrypoint finding independently of the active
    CausalLM-vs-CausalLM SPEC-01 comparison.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    rope = Qwen3_5TextRotaryEmbedding(config=text_config)
    hidden = torch.zeros(1, 1, text_config.hidden_size)

    for position in (0, 1, 7, 1024):
        three_axes = torch.full((3, 1, 1), position, dtype=torch.long)
        one_axis = torch.full((1, 1, 1), position, dtype=torch.long)
        cos3, sin3 = rope(hidden, three_axes)
        cos1, sin1 = rope(hidden, one_axis)
        assert cos1.shape == cos3.shape
        assert torch.equal(cos1, cos3), f"cos differs at position {position}"
        assert torch.equal(sin1, sin3), f"sin differs at position {position}"


def test_rotary_output_width_is_the_partial_rope_dimension(text_config):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    rope = Qwen3_5TextRotaryEmbedding(config=text_config)
    hidden = torch.zeros(1, 3, text_config.hidden_size)
    cos, sin = rope(hidden, torch.zeros(3, 1, 3, dtype=torch.long))
    # Partial RoPE: only 64 of the 256 head dimensions are rotated (audit 07).
    assert cos.shape[-1] == C.ROTARY_DIM == 64
    assert sin.shape[-1] == C.ROTARY_DIM


def test_only_the_multimodal_entry_point_overrides_position_id_preparation():
    """Formic's entry point must stay on the generic implementation."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    assert "_prepare_position_ids_for_generation" in vars(Qwen3_5ForConditionalGeneration), (
        "the multimodal class no longer overrides position-id preparation; the step-1 "
        "finding (report section 5.6) needs re-checking"
    )
    assert "_prepare_position_ids_for_generation" not in vars(Qwen3_5ForCausalLM), (
        "the text-only class now overrides position-id preparation; Formic's decode "
        "path assumptions must be re-validated"
    )


def test_text_only_class_builds_no_vision_tower(text_config):
    """A7 at the source level: the class Formic uses cannot construct a tower."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    source = inspect.getsource(Qwen3_5ForCausalLM.__init__)
    assert "Qwen3_5TextModel" in source
    assert "Vision" not in source and "visual" not in source
