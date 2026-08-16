"""Audited constants of the Qwen3.8-27B checkpoint.

Every value in this module is an AUDIT FACT taken from
``/workspace/audits/qwen3_8_27b/`` (reports 01-15 + FINAL_AUDIT_REPORT) and
re-verified against the local checkpoint. Nothing here may be "adjusted" to make
code pass: a mismatch between these constants and the loaded checkpoint means the
checkpoint is not the audited one, and execution must stop.

Audit constraint registry references (see plan ``formic_plan_implementation_initial``
section 3) are quoted where a constant encodes one.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

PRODUCT_NAME: Final[str] = "Qwen3.8-27B"
#: The product is named Qwen3.8 but the runtime architecture is Qwen3.5 (audit 01/02).
RUNTIME_ARCHITECTURE: Final[str] = "Qwen3_5ForConditionalGeneration"
RUNTIME_MODEL_TYPE: Final[str] = "qwen3_5"
TEXT_MODEL_TYPE: Final[str] = "qwen3_5_text"
CHECKPOINT_COMMIT: Final[str] = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TRANSFORMERS_VERSION_AUDITED: Final[str] = "5.8.0"

# --------------------------------------------------------------------------
# Text backbone geometry
# --------------------------------------------------------------------------

NUM_LAYERS: Final[int] = 64
HIDDEN_SIZE: Final[int] = 5_120
INTERMEDIATE_SIZE: Final[int] = 17_408
VOCAB_SIZE: Final[int] = 248_320
MAX_POSITION_EMBEDDINGS: Final[int] = 262_144
RMS_NORM_EPS: Final[float] = 1e-6
TIE_WORD_EMBEDDINGS: Final[bool] = False
HIDDEN_ACT: Final[str] = "silu"

# --------------------------------------------------------------------------
# Hybrid group structure (A11)
# --------------------------------------------------------------------------

GROUP_SIZE: Final[int] = 4
NUM_GROUPS: Final[int] = 16
#: Layer type of each position inside a hybrid group.
GROUP_PATTERN: Final[tuple[str, ...]] = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
)
LINEAR_ATTENTION_TYPE: Final[str] = "linear_attention"
FULL_ATTENTION_TYPE: Final[str] = "full_attention"
#: 0-indexed indices of the Full Attention layers: 3, 7, ..., 63.
ATTENTION_LAYER_INDICES: Final[tuple[int, ...]] = tuple(range(3, NUM_LAYERS, GROUP_SIZE))
GDN_LAYER_INDICES: Final[tuple[int, ...]] = tuple(
    i for i in range(NUM_LAYERS) if i not in ATTENTION_LAYER_INDICES
)
NUM_ATTENTION_LAYERS: Final[int] = 16
NUM_GDN_LAYERS: Final[int] = 48
#: ``get_seq_length()`` reads the first full-attention layer, index 3 (audit 08).
SEQ_LENGTH_ANCHOR_LAYER: Final[int] = 3
#: Number of insertion points: before G1, between each Gi/Gi+1, after G16.
NUM_BOUNDARIES: Final[int] = NUM_GROUPS + 1

# --------------------------------------------------------------------------
# Full Attention (audit 07)
# --------------------------------------------------------------------------

NUM_ATTENTION_HEADS: Final[int] = 24
NUM_KEY_VALUE_HEADS: Final[int] = 4
HEAD_DIM: Final[int] = 256
GQA_GROUPS: Final[int] = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
#: ``q_proj`` is double width: 6144 query values + 6144 gate values.
Q_PROJ_OUT_FEATURES: Final[int] = 12_288
KV_PROJ_OUT_FEATURES: Final[int] = 1_024
O_PROJ_IN_FEATURES: Final[int] = 6_144
ROPE_THETA: Final[float] = 10_000_000.0
PARTIAL_ROTARY_FACTOR: Final[float] = 0.25
#: 64 of the 256 head dimensions are rotated; 192 pass through unrotated.
ROTARY_DIM: Final[int] = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
MROPE_SECTION: Final[tuple[int, int, int]] = (11, 11, 10)
MROPE_INTERLEAVED: Final[bool] = True

# --------------------------------------------------------------------------
# Gated DeltaNet (audit 06)
# --------------------------------------------------------------------------

LINEAR_NUM_KEY_HEADS: Final[int] = 16
LINEAR_NUM_VALUE_HEADS: Final[int] = 48
LINEAR_KEY_HEAD_DIM: Final[int] = 128
LINEAR_VALUE_HEAD_DIM: Final[int] = 128
LINEAR_CONV_KERNEL_DIM: Final[int] = 4
#: mixed QKV width = 2*2048 (Q,K) + 6144 (V).
MIXED_QKV_WIDTH: Final[int] = 10_240
LINEAR_Z_WIDTH: Final[int] = 6_144
#: Q/K heads are repeated 3x to align with the 48 value heads.
QK_HEAD_REPEAT: Final[int] = LINEAR_NUM_VALUE_HEADS // LINEAR_NUM_KEY_HEADS
#: Persistent state shapes, batch dimension excluded.
GDN_CONV_STATE_SHAPE: Final[tuple[int, int]] = (MIXED_QKV_WIDTH, LINEAR_CONV_KERNEL_DIM)
GDN_RECURRENT_STATE_SHAPE: Final[tuple[int, int, int]] = (
    LINEAR_NUM_VALUE_HEADS,
    LINEAR_KEY_HEAD_DIM,
    LINEAR_VALUE_HEAD_DIM,
)
#: Config field ``mamba_ssm_dtype="float32"`` is NOT consumed by the runtime:
#: the recurrence computes in FP32 but the persistent state is BF16 (audit 02/06).
#: This is the source of the segmentation rounding measured in step 2 (E4).
MAMBA_SSM_DTYPE_DECLARED: Final[str] = "float32"
GDN_PERSISTENT_STATE_DTYPE_OBSERVED: Final[str] = "bfloat16"

# --------------------------------------------------------------------------
# Cache memory constants, BF16, batch 1 (audit 13)
# --------------------------------------------------------------------------

GDN_CONV_STATE_BYTES_PER_LAYER: Final[int] = 81_920
GDN_RECURRENT_STATE_BYTES_PER_LAYER: Final[int] = 1_572_864
GDN_STATE_BYTES_PER_LAYER: Final[int] = 1_654_784
GDN_STATE_BYTES_TOTAL: Final[int] = 79_429_632
KV_BYTES_PER_TOKEN_PER_LAYER: Final[int] = 4_096
KV_BYTES_PER_TOKEN: Final[int] = 65_536

# --------------------------------------------------------------------------
# Parameter and tensor inventory (audit 01/03/09)
# --------------------------------------------------------------------------

TOTAL_TENSORS: Final[int] = 1_199
TOTAL_STORED_PARAMS: Final[int] = 27_781_427_952
TOTAL_PAYLOAD_BYTES: Final[int] = 55_562_855_904
PARAMS_LOADED_BY_TRANSFORMERS: Final[int] = 27_356_728_560
MTP_PARAMS: Final[int] = 424_699_392
MTP_TENSORS: Final[int] = 15
VISION_PARAMS: Final[int] = 460_730_096
VISION_TENSORS: Final[int] = 333
EMBEDDING_PARAMS: Final[int] = 1_271_398_400
LM_HEAD_PARAMS: Final[int] = 1_271_398_400
NUM_SHARDS: Final[int] = 18
PARAMS_PER_GDN_LAYER: Final[int] = 383_273_184
PARAMS_PER_ATTENTION_LAYER: Final[int] = 372_255_232
PARAMS_PER_GROUP: Final[int] = 1_522_074_784
CHECKPOINT_DTYPE: Final[str] = "bfloat16"

# --------------------------------------------------------------------------
# Tokenizer (audit 01)
# --------------------------------------------------------------------------

TOKENIZER_EFFECTIVE_SIZE: Final[int] = 248_077
#: 243 embedding/LM-head rows exist with no tokenizer mapping. They are
#: shape-present but untrained: reserved for control tokens introduced in step 8,
#: never usable by a frozen model (plan section 2.3).
RESERVED_ROW_START: Final[int] = 248_077
RESERVED_ROW_END_EXCLUSIVE: Final[int] = 248_320
NUM_RESERVED_ROWS: Final[int] = RESERVED_ROW_END_EXCLUSIVE - RESERVED_ROW_START

BOS_TOKEN_ID: Final[int] = 248_044
PAD_TOKEN_ID: Final[int] = 248_044
TOKENIZER_EOS_TOKEN_ID: Final[int] = 248_046
GENERATION_EOS_TOKEN_IDS: Final[tuple[int, int]] = (248_046, 248_044)

VISION_START_TOKEN_ID: Final[int] = 248_053
VISION_END_TOKEN_ID: Final[int] = 248_054
VISION_PAD_TOKEN_ID: Final[int] = 248_055
IMAGE_TOKEN_ID: Final[int] = 248_056
VIDEO_TOKEN_ID: Final[int] = 248_057

# --------------------------------------------------------------------------
# Checkpoint generation defaults (audit 02) - payload-class only, never control
# --------------------------------------------------------------------------

CHECKPOINT_DEFAULT_TEMPERATURE: Final[float] = 1.0
CHECKPOINT_DEFAULT_TOP_P: Final[float] = 0.95
CHECKPOINT_DEFAULT_TOP_K: Final[int] = 20

# --------------------------------------------------------------------------
# Tensor namespaces
# --------------------------------------------------------------------------

#: Prefix used by the checkpoint for text weights (``Qwen3_5ForConditionalGeneration``).
CKPT_TEXT_PREFIX: Final[str] = "model.language_model."
#: Prefix expected by ``Qwen3_5ForCausalLM`` (text-only path, A7).
CAUSAL_LM_TEXT_PREFIX: Final[str] = "model."
CKPT_VISION_PREFIX: Final[str] = "model.visual."
CKPT_MTP_PREFIX: Final[str] = "mtp."
CKPT_LM_HEAD_KEY: Final[str] = "lm_head.weight"


def group_index_of_layer(layer_index: int) -> int:
    """Return the 1-based hybrid group index containing ``layer_index`` (0-based)."""
    if not 0 <= layer_index < NUM_LAYERS:
        raise ValueError(f"layer_index out of range: {layer_index}")
    return layer_index // GROUP_SIZE + 1


def layers_of_group(group_index: int) -> tuple[int, ...]:
    """Return the 0-based layer indices of the 1-based hybrid group ``group_index``."""
    if not 1 <= group_index <= NUM_GROUPS:
        raise ValueError(f"group_index out of range: {group_index}")
    start = (group_index - 1) * GROUP_SIZE
    return tuple(range(start, start + GROUP_SIZE))


def expected_layer_types() -> tuple[str, ...]:
    """Return the expected 64-entry layer-type list: 16 x (GDN, GDN, GDN, Attention)."""
    return GROUP_PATTERN * NUM_GROUPS


def kv_cache_bytes(num_tokens: int, num_attention_layers: int = NUM_ATTENTION_LAYERS) -> int:
    """Exact BF16 KV bytes for ``num_tokens`` at batch 1 (audit 13)."""
    return num_tokens * KV_BYTES_PER_TOKEN_PER_LAYER * num_attention_layers


def gdn_state_bytes(num_gdn_layers: int = NUM_GDN_LAYERS) -> int:
    """Exact BF16 GDN state bytes at batch 1 (audit 13)."""
    return num_gdn_layers * GDN_STATE_BYTES_PER_LAYER


def total_cache_bytes(
    num_tokens: int,
    num_gdn_layers: int = NUM_GDN_LAYERS,
    num_attention_layers: int = NUM_ATTENTION_LAYERS,
) -> int:
    """Total hybrid cache bytes at batch 1 for a given active-layer subset."""
    return gdn_state_bytes(num_gdn_layers) + kv_cache_bytes(num_tokens, num_attention_layers)
