"""Strict tensor inventory of the checkpoint (audit constraint A12).

*Permissive loading is forbidden.* Every tensor of the checkpoint is accounted
for before any weight is read, and every tensor of the loaded model is matched
back to the checkpoint afterwards. A missing tensor, an unexpected tensor, a
shape mismatch or a dtype mismatch is FATAL — never a warning.

Exclusions are **declared**, never silent: in text-only mode the vision tensors
(333) and the MTP tensors (15, A10) are excluded on purpose, and their exact
counts are verified. Hugging Face's own ``_keys_to_ignore_on_load_unexpected``
would hide them; this module makes them visible and checked.

Reading uses the safetensors headers only — no weight bytes are touched.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from formic.backbone import constants as C

__all__ = [
    "TensorRecord",
    "CheckpointInventory",
    "StrictLoadReport",
    "InventoryError",
    "assert_strict_load",
]

Family = Literal["text", "lm_head", "vision", "mtp"]
BackboneMode = Literal["text_only", "reference_multimodal"]

#: safetensors dtype string -> (torch dtype name, bytes per element)
_DTYPES = {
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}


class InventoryError(RuntimeError):
    """Any inventory violation. Always fatal."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str  # torch dtype name
    num_params: int
    shard: str
    family: Family

    @property
    def nbytes(self) -> int:
        return self.num_params * _DTYPES_BY_TORCH_NAME[self.dtype]


_DTYPES_BY_TORCH_NAME = {torch_name: size for torch_name, size in _DTYPES.values()}


def classify(name: str) -> Family:
    """Assign a checkpoint tensor to its family from its name alone."""
    if name == C.CKPT_LM_HEAD_KEY:
        return "lm_head"
    if name.startswith(C.CKPT_MTP_PREFIX):
        return "mtp"
    if name.startswith(C.CKPT_VISION_PREFIX):
        return "vision"
    if name.startswith(C.CKPT_TEXT_PREFIX):
        return "text"
    raise InventoryError(
        f"tensor {name!r} matches no known family; the checkpoint is not the audited one"
    )


@dataclass
class CheckpointInventory:
    """Complete, header-derived inventory of a Qwen3.8-27B checkpoint directory."""

    path: Path
    records: tuple[TensorRecord, ...]
    index_map: dict[str, str] = field(repr=False, default_factory=dict)
    index_total_size: int | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CheckpointInventory":
        path = Path(path)
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise InventoryError(f"missing safetensors index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        index_total = index.get("metadata", {}).get("total_size")

        records: list[TensorRecord] = []
        seen: set[str] = set()
        for shard in sorted(set(weight_map.values())):
            shard_path = path / shard
            if not shard_path.is_file():
                raise InventoryError(f"shard declared in index but missing on disk: {shard}")
            for name, meta in _read_safetensors_header(shard_path).items():
                if name == "__metadata__":
                    continue
                declared_shard = weight_map.get(name)
                if declared_shard is None:
                    raise InventoryError(f"tensor {name!r} present in {shard} but absent from the index")
                if declared_shard != shard:
                    raise InventoryError(
                        f"tensor {name!r} found in {shard} but index declares {declared_shard}"
                    )
                if name in seen:
                    raise InventoryError(f"tensor {name!r} appears in more than one shard")
                seen.add(name)
                dtype_name, _ = _DTYPES[meta["dtype"]]
                shape = tuple(int(d) for d in meta["shape"])
                records.append(
                    TensorRecord(
                        name=name,
                        shape=shape,
                        dtype=dtype_name,
                        num_params=_numel(shape),
                        shard=shard,
                        family=classify(name),
                    )
                )

        missing = sorted(set(weight_map) - seen)
        if missing:
            raise InventoryError(f"{len(missing)} tensor(s) declared in the index but not found: {missing[:5]}")

        return cls(
            path=path,
            records=tuple(sorted(records, key=lambda r: r.name)),
            index_map=weight_map,
            index_total_size=index_total,
        )

    # -- accessors ---------------------------------------------------------

    def by_family(self, family: Family) -> tuple[TensorRecord, ...]:
        return tuple(r for r in self.records if r.family == family)

    def params(self, family: Family | None = None) -> int:
        source: Iterable[TensorRecord] = self.records if family is None else self.by_family(family)
        return sum(r.num_params for r in source)

    def total_bytes(self) -> int:
        return sum(r.nbytes for r in self.records)

    def shards(self) -> tuple[str, ...]:
        return tuple(sorted(set(r.shard for r in self.records)))

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.path),
            "num_tensors": len(self.records),
            "num_shards": len(self.shards()),
            "total_params": self.params(),
            "total_bytes": self.total_bytes(),
            "index_total_size": self.index_total_size,
            "families": {
                family: {
                    "tensors": len(self.by_family(family)),  # type: ignore[arg-type]
                    "params": self.params(family),  # type: ignore[arg-type]
                }
                for family in ("text", "lm_head", "vision", "mtp")
            },
            "dtypes": sorted({r.dtype for r in self.records}),
        }

    # -- audit conformance -------------------------------------------------

    def validate_against_audit(self) -> None:
        """Check the inventory against the audited facts. Raises on any divergence."""
        problems: list[str] = []

        if len(self.records) != C.TOTAL_TENSORS:
            problems.append(f"tensor count {len(self.records)} != {C.TOTAL_TENSORS}")
        if self.params() != C.TOTAL_STORED_PARAMS:
            problems.append(f"stored params {self.params()} != {C.TOTAL_STORED_PARAMS}")
        if self.total_bytes() != C.TOTAL_PAYLOAD_BYTES:
            problems.append(f"payload bytes {self.total_bytes()} != {C.TOTAL_PAYLOAD_BYTES}")
        if self.index_total_size is not None and self.index_total_size != C.TOTAL_PAYLOAD_BYTES:
            problems.append(
                f"index metadata.total_size {self.index_total_size} != {C.TOTAL_PAYLOAD_BYTES}"
            )
        if len(self.shards()) != C.NUM_SHARDS:
            problems.append(f"shard count {len(self.shards())} != {C.NUM_SHARDS}")

        dtypes = {r.dtype for r in self.records}
        if dtypes != {C.CHECKPOINT_DTYPE}:
            problems.append(f"dtypes {sorted(dtypes)} != ['{C.CHECKPOINT_DTYPE}']")

        expected_counts = {
            "vision": C.VISION_TENSORS,
            "mtp": C.MTP_TENSORS,
            "lm_head": 1,
        }
        for family, expected in expected_counts.items():
            actual = len(self.by_family(family))  # type: ignore[arg-type]
            if actual != expected:
                problems.append(f"{family} tensor count {actual} != {expected}")

        expected_params = {
            "vision": C.VISION_PARAMS,
            "mtp": C.MTP_PARAMS,
            "lm_head": C.LM_HEAD_PARAMS,
        }
        for family, expected in expected_params.items():
            actual = self.params(family)  # type: ignore[arg-type]
            if actual != expected:
                problems.append(f"{family} params {actual} != {expected}")

        # Text layer coverage: exactly 64 layers must be present.
        layer_indices = set()
        for record in self.by_family("text"):
            suffix = record.name[len(C.CKPT_TEXT_PREFIX) :]
            if suffix.startswith("layers."):
                layer_indices.add(int(suffix.split(".")[1]))
        if layer_indices != set(range(C.NUM_LAYERS)):
            problems.append(
                f"text layer indices cover {len(layer_indices)} layers, expected {C.NUM_LAYERS}"
            )

        # The mixer family of each layer must match the audited hybrid pattern.
        for index in sorted(layer_indices):
            prefix = f"{C.CKPT_TEXT_PREFIX}layers.{index}."
            has_gdn = any(r.name.startswith(prefix + "linear_attn.") for r in self.by_family("text"))
            has_attn = any(r.name.startswith(prefix + "self_attn.") for r in self.by_family("text"))
            expects_attn = index in C.ATTENTION_LAYER_INDICES
            if expects_attn and not (has_attn and not has_gdn):
                problems.append(f"layer {index} should be full_attention by weight names")
            if not expects_attn and not (has_gdn and not has_attn):
                problems.append(f"layer {index} should be linear_attention by weight names")

        embed = self._get(f"{C.CKPT_TEXT_PREFIX}embed_tokens.weight")
        head = self._get(C.CKPT_LM_HEAD_KEY)
        for record, label in ((embed, "embed_tokens"), (head, "lm_head")):
            if record.shape != (C.VOCAB_SIZE, C.HIDDEN_SIZE):
                problems.append(f"{label} shape {record.shape} != {(C.VOCAB_SIZE, C.HIDDEN_SIZE)}")

        if problems:
            raise InventoryError(
                "checkpoint does not match the audited facts:\n  - " + "\n  - ".join(problems)
            )

    def _get(self, name: str) -> TensorRecord:
        for record in self.records:
            if record.name == name:
                return record
        raise InventoryError(f"required tensor missing from checkpoint: {name}")

    # -- expectations for a loaded model -----------------------------------

    def expected_model_tensors(self, mode: BackboneMode) -> dict[str, tuple[int, ...]]:
        """Map ``loaded parameter name -> shape`` for a given backbone mode.

        ``text_only``
            ``Qwen3_5ForCausalLM``: checkpoint ``model.language_model.*`` becomes
            ``model.*`` (A7 — the vision tower is never constructed).
        ``reference_multimodal``
            stock ``Qwen3_5ForConditionalGeneration``: names unchanged, vision
            included, MTP still excluded (A10).
        """
        expected: dict[str, tuple[int, ...]] = {}
        for record in self.records:
            if record.family == "mtp":
                continue  # A10: never loaded in part 1
            if record.family == "vision" and mode == "text_only":
                continue  # A7: tower not constructed
            expected[remap_key(record.name, mode)] = record.shape
        return expected

    def declared_exclusions(self, mode: BackboneMode) -> dict[str, int]:
        """Tensor counts intentionally excluded, by family. Verified, never silent."""
        exclusions = {"mtp": len(self.by_family("mtp"))}
        if mode == "text_only":
            exclusions["vision"] = len(self.by_family("vision"))
        return exclusions

    def key_mapping(self, mode: BackboneMode) -> dict[str, str]:
        """Regex key mapping handed to ``from_pretrained(key_mapping=...)``.

        Pure renaming of the text namespace; no tensor is split, merged,
        transposed or given a new semantic role.
        """
        if mode != "text_only":
            return {}
        return {r"^model\.language_model\.": "model."}


def remap_key(checkpoint_name: str, mode: BackboneMode) -> str:
    """Translate a checkpoint tensor name into the loaded-model namespace."""
    if mode == "text_only" and checkpoint_name.startswith(C.CKPT_TEXT_PREFIX):
        return C.CAUSAL_LM_TEXT_PREFIX + checkpoint_name[len(C.CKPT_TEXT_PREFIX) :]
    return checkpoint_name


# --------------------------------------------------------------------------
# Post-load verification
# --------------------------------------------------------------------------


@dataclass
class StrictLoadReport:
    """Result of matching a loaded model against the checkpoint inventory."""

    mode: BackboneMode
    matched: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    dtype_mismatch: tuple[tuple[str, str, str], ...]
    declared_exclusions: dict[str, int]
    loaded_params: int
    expected_params: int
    vision_tower_present: bool
    mtp_module_present: bool
    embeddings_tied: bool

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatch or self.dtype_mismatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "matched": self.matched,
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatch": [list(x) for x in self.shape_mismatch],
            "dtype_mismatch": [list(x) for x in self.dtype_mismatch],
            "declared_exclusions": self.declared_exclusions,
            "loaded_params": self.loaded_params,
            "expected_params": self.expected_params,
            "vision_tower_present": self.vision_tower_present,
            "mtp_module_present": self.mtp_module_present,
            "embeddings_tied": self.embeddings_tied,
        }

    def render(self) -> str:
        lines = [
            f"STRICT LOAD REPORT [{self.mode}]",
            f"  matched tensors      : {self.matched}",
            f"  loaded params        : {self.loaded_params:,}",
            f"  expected params      : {self.expected_params:,}",
            f"  declared exclusions  : {self.declared_exclusions}",
            f"  vision tower present : {self.vision_tower_present}",
            f"  mtp module present   : {self.mtp_module_present}",
            f"  embeddings tied      : {self.embeddings_tied}",
            f"  missing              : {len(self.missing)}",
            f"  unexpected           : {len(self.unexpected)}",
            f"  shape mismatches     : {len(self.shape_mismatch)}",
            f"  dtype mismatches     : {len(self.dtype_mismatch)}",
            f"  RESULT               : {'PASS' if self.ok else 'FAIL'}",
        ]
        for name in self.missing[:10]:
            lines.append(f"    missing: {name}")
        for name in self.unexpected[:10]:
            lines.append(f"    unexpected: {name}")
        return "\n".join(lines)


def assert_strict_load(
    model: Any,
    inventory: CheckpointInventory,
    mode: BackboneMode,
    *,
    raise_on_failure: bool = True,
) -> StrictLoadReport:
    """Verify a loaded model against the checkpoint inventory (A12).

    Compares parameter names, shapes and dtypes both ways. Also asserts the
    structural expectations that make part-1 rules checkable: no vision tower in
    text-only mode (A7), no MTP module (A10), untied embeddings.
    """
    expected = inventory.expected_model_tensors(mode)
    actual: dict[str, Any] = dict(model.named_parameters())

    missing = tuple(sorted(set(expected) - set(actual)))
    unexpected = tuple(sorted(set(actual) - set(expected)))

    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    dtype_mismatch: list[tuple[str, str, str]] = []
    for name in sorted(set(expected) & set(actual)):
        param = actual[name]
        want = expected[name]
        got = tuple(param.shape)
        if got != want:
            shape_mismatch.append((name, got, want))
        dtype_name = str(param.dtype).replace("torch.", "")
        if dtype_name != C.CHECKPOINT_DTYPE:
            dtype_mismatch.append((name, dtype_name, C.CHECKPOINT_DTYPE))

    text_model = getattr(getattr(model, "model", None), "language_model", None)
    vision_present = hasattr(getattr(model, "model", None), "visual") or (
        text_model is not None and hasattr(model.model, "visual")
    )
    mtp_present = any("mtp" in name.split(".") for name in actual)

    embeddings_tied = _detect_tied_embeddings(model)

    report = StrictLoadReport(
        mode=mode,
        matched=len(set(expected) & set(actual)) - len(shape_mismatch),
        missing=missing,
        unexpected=unexpected,
        shape_mismatch=tuple(shape_mismatch),
        dtype_mismatch=tuple(dtype_mismatch),
        declared_exclusions=inventory.declared_exclusions(mode),
        loaded_params=sum(p.numel() for p in actual.values()),
        expected_params=sum(_numel(shape) for shape in expected.values()),
        vision_tower_present=bool(vision_present),
        mtp_module_present=mtp_present,
        embeddings_tied=embeddings_tied,
    )

    if raise_on_failure:
        problems: list[str] = []
        if not report.ok:
            problems.append("tensor set does not match the checkpoint inventory")
        if report.loaded_params != report.expected_params:
            problems.append(
                f"loaded params {report.loaded_params:,} != expected {report.expected_params:,}"
            )
        if mode == "text_only" and report.vision_tower_present:
            problems.append("vision tower is present in text-only mode (violates A7)")
        if report.mtp_module_present:
            problems.append("an MTP module is present (violates A10 for part 1)")
        if report.embeddings_tied != C.TIE_WORD_EMBEDDINGS:
            problems.append(
                f"embeddings_tied={report.embeddings_tied}, checkpoint declares "
                f"tie_word_embeddings={C.TIE_WORD_EMBEDDINGS}"
            )
        if problems:
            raise InventoryError(report.render() + "\n\nFATAL:\n  - " + "\n  - ".join(problems))
    return report


def _detect_tied_embeddings(model: Any) -> bool:
    """Whether input embeddings and LM head share storage.

    Identity of the ``Parameter`` object is the reliable signal: Hugging Face
    ties by assigning the same object. ``data_ptr()`` is only used as a
    secondary check on *materialised* tensors — with CPU offload, offloaded
    parameters live on the ``meta`` device where every ``data_ptr()`` is 0, so a
    pointer comparison alone would report every model as tied.
    """
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
    except Exception:  # pragma: no cover - model without the accessors
        return False
    if input_embeddings is None or output_embeddings is None:
        return False

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if input_weight is None or output_weight is None:
        return False
    if input_weight is output_weight:
        return True

    if input_weight.device.type == "meta" or output_weight.device.type == "meta":
        return False  # storage-level comparison is meaningless on meta
    input_ptr, output_ptr = input_weight.data_ptr(), output_weight.data_ptr()
    return bool(input_ptr and output_ptr and input_ptr == output_ptr)


# --------------------------------------------------------------------------
# safetensors header reading (no weight bytes are read)
# --------------------------------------------------------------------------


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise InventoryError(f"truncated safetensors file: {path}")
        (length,) = struct.unpack("<Q", raw_length)
        header = handle.read(length)
        if len(header) != length:
            raise InventoryError(f"truncated safetensors header: {path}")
    return json.loads(header.decode("utf-8"))


def _numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
