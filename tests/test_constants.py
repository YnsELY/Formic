"""Audited constants must be internally consistent.

If any of these fail, the constants file was edited without re-reading the
audit, and every downstream number becomes unreliable.
"""

from __future__ import annotations

from formic.backbone import constants as C


def test_layer_and_group_counts():
    assert C.NUM_LAYERS == 64
    assert C.NUM_GROUPS == 16
    assert C.GROUP_SIZE == 4
    assert C.NUM_GROUPS * C.GROUP_SIZE == C.NUM_LAYERS
    assert C.NUM_BOUNDARIES == 17


def test_attention_layer_indices_are_the_audited_ones():
    assert C.ATTENTION_LAYER_INDICES == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
    assert len(C.ATTENTION_LAYER_INDICES) == C.NUM_ATTENTION_LAYERS == 16
    assert len(C.GDN_LAYER_INDICES) == C.NUM_GDN_LAYERS == 48
    assert set(C.ATTENTION_LAYER_INDICES) & set(C.GDN_LAYER_INDICES) == set()
    assert len(set(C.ATTENTION_LAYER_INDICES) | set(C.GDN_LAYER_INDICES)) == C.NUM_LAYERS


def test_every_group_ends_on_a_full_attention_layer():
    """Each exit sits immediately after a globally-mixing attention layer."""
    for group in range(1, C.NUM_GROUPS + 1):
        layers = C.layers_of_group(group)
        assert layers[-1] in C.ATTENTION_LAYER_INDICES
        assert all(layer not in C.ATTENTION_LAYER_INDICES for layer in layers[:-1])


def test_expected_layer_types_pattern():
    types = C.expected_layer_types()
    assert len(types) == C.NUM_LAYERS
    assert types[:4] == C.GROUP_PATTERN
    attention = tuple(i for i, t in enumerate(types) if t == C.FULL_ATTENTION_TYPE)
    assert attention == C.ATTENTION_LAYER_INDICES


def test_group_membership_helpers_roundtrip():
    for layer in range(C.NUM_LAYERS):
        group = C.group_index_of_layer(layer)
        assert layer in C.layers_of_group(group)
    assert C.group_index_of_layer(0) == 1
    assert C.group_index_of_layer(63) == 16
    assert C.layers_of_group(1) == (0, 1, 2, 3)
    assert C.layers_of_group(16) == (60, 61, 62, 63)


def test_seq_length_anchor_is_first_attention_layer():
    assert C.SEQ_LENGTH_ANCHOR_LAYER == min(C.ATTENTION_LAYER_INDICES) == 3
    assert C.SEQ_LENGTH_ANCHOR_LAYER < C.GROUP_SIZE  # inside group 1: active on every route


def test_parameter_arithmetic_matches_audit():
    per_group = 3 * C.PARAMS_PER_GDN_LAYER + C.PARAMS_PER_ATTENTION_LAYER
    assert per_group == C.PARAMS_PER_GROUP == 1_522_074_784
    assert C.TOTAL_STORED_PARAMS - C.MTP_PARAMS == C.PARAMS_LOADED_BY_TRANSFORMERS
    assert C.EMBEDDING_PARAMS == C.LM_HEAD_PARAMS == C.VOCAB_SIZE * C.HIDDEN_SIZE


def test_cache_byte_formulas_match_audit_13():
    assert C.GDN_CONV_STATE_BYTES_PER_LAYER == C.MIXED_QKV_WIDTH * C.LINEAR_CONV_KERNEL_DIM * 2
    recurrent = (
        C.LINEAR_NUM_VALUE_HEADS * C.LINEAR_KEY_HEAD_DIM * C.LINEAR_VALUE_HEAD_DIM * 2
    )
    assert C.GDN_RECURRENT_STATE_BYTES_PER_LAYER == recurrent == 1_572_864
    assert (
        C.GDN_STATE_BYTES_PER_LAYER
        == C.GDN_CONV_STATE_BYTES_PER_LAYER + C.GDN_RECURRENT_STATE_BYTES_PER_LAYER
    )
    assert C.gdn_state_bytes() == C.GDN_STATE_BYTES_TOTAL == 79_429_632

    per_token_per_layer = 2 * C.NUM_KEY_VALUE_HEADS * C.HEAD_DIM * 2  # K and V, BF16
    assert per_token_per_layer == C.KV_BYTES_PER_TOKEN_PER_LAYER == 4_096
    assert C.KV_BYTES_PER_TOKEN == C.KV_BYTES_PER_TOKEN_PER_LAYER * C.NUM_ATTENTION_LAYERS

    # Audit 13 reference points.
    assert C.total_cache_bytes(4) == 79_691_776
    assert C.total_cache_bytes(8) == 79_953_920
    assert C.total_cache_bytes(1_024) == 146_538_496
    assert C.total_cache_bytes(262_144) == 17_259_298_816


def test_gdn_head_geometry():
    assert C.QK_HEAD_REPEAT == 3
    assert C.LINEAR_NUM_KEY_HEADS * C.QK_HEAD_REPEAT == C.LINEAR_NUM_VALUE_HEADS
    qk_width = C.LINEAR_NUM_KEY_HEADS * C.LINEAR_KEY_HEAD_DIM
    v_width = C.LINEAR_NUM_VALUE_HEADS * C.LINEAR_VALUE_HEAD_DIM
    assert 2 * qk_width + v_width == C.MIXED_QKV_WIDTH == 10_240
    assert v_width == C.LINEAR_Z_WIDTH == 6_144


def test_attention_geometry():
    assert C.NUM_ATTENTION_HEADS * C.HEAD_DIM == 6_144
    assert C.Q_PROJ_OUT_FEATURES == 2 * C.NUM_ATTENTION_HEADS * C.HEAD_DIM  # q + output gate
    assert C.KV_PROJ_OUT_FEATURES == C.NUM_KEY_VALUE_HEADS * C.HEAD_DIM
    assert C.O_PROJ_IN_FEATURES == C.NUM_ATTENTION_HEADS * C.HEAD_DIM
    assert C.GQA_GROUPS == 6
    assert C.ROTARY_DIM == 64 and C.ROTARY_DIM < C.HEAD_DIM  # partial RoPE
    assert sum(C.MROPE_SECTION) == C.ROTARY_DIM // 2


def test_reserved_vocabulary_rows():
    assert C.NUM_RESERVED_ROWS == 243
    assert C.RESERVED_ROW_START == C.TOKENIZER_EFFECTIVE_SIZE
    assert C.RESERVED_ROW_END_EXCLUSIVE == C.VOCAB_SIZE
    assert C.VOCAB_SIZE % 256 == 0
