"""Hybrid group view over the intact Hugging Face decoder stack.

Audit constraint A11: the 16 hybrid groups are a **partition / view** over the
untouched HF modules. Nothing here re-implements a cell, copies cell code, or
changes the forward graph. The view only *names* things:

* which layers belong to which group,
* which layers are Gated DeltaNet and which are Full Attention,
* where the 17 group boundaries sit in the residual stream.

Everything is cross-checked against :mod:`formic.backbone.constants` (audit
facts) and, when a model is available, against the real instantiated modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from formic.backbone import constants as C

__all__ = [
    "LayerSpec",
    "GroupSpec",
    "Boundary",
    "HybridGroupView",
    "BOUNDARY_NAMES",
    "GroupStructureError",
    "boundary_names",
]


class GroupStructureError(RuntimeError):
    """Raised when the checkpoint/model does not match the audited structure."""


def boundary_names() -> tuple[str, ...]:
    """The 17 boundary names: ``PRE_G1``, ``G1_G2`` ... ``G15_G16``, ``POST_G16``."""
    names = ["PRE_G1"]
    names += [f"G{i}_G{i + 1}" for i in range(1, C.NUM_GROUPS)]
    names.append(f"POST_G{C.NUM_GROUPS}")
    return tuple(names)


#: Canonical, ordered boundary names (17 entries).
BOUNDARY_NAMES: tuple[str, ...] = boundary_names()


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer, described by position rather than by re-implementation."""

    index: int
    layer_type: str
    group_index: int
    position_in_group: int

    @property
    def is_attention(self) -> bool:
        return self.layer_type == C.FULL_ATTENTION_TYPE

    @property
    def is_gdn(self) -> bool:
        return self.layer_type == C.LINEAR_ATTENTION_TYPE

    @property
    def mixer_attr(self) -> str:
        """Attribute name of the token mixer on the HF decoder layer."""
        return "self_attn" if self.is_attention else "linear_attn"


@dataclass(frozen=True)
class GroupSpec:
    """One hybrid group: ``[GDN, GDN, GDN, Full Attention]``."""

    index: int  # 1-based
    layer_indices: tuple[int, ...]
    layer_types: tuple[str, ...]

    @property
    def first_layer(self) -> int:
        return self.layer_indices[0]

    @property
    def last_layer(self) -> int:
        return self.layer_indices[-1]

    @property
    def attention_layer(self) -> int:
        """The group's single Full Attention layer (its global-mixing anchor)."""
        return self.layer_indices[C.GROUP_SIZE - 1]

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return self.layer_indices[: C.GROUP_SIZE - 1]

    @property
    def entry_boundary(self) -> str:
        return "PRE_G1" if self.index == 1 else f"G{self.index - 1}_G{self.index}"

    @property
    def exit_boundary(self) -> str:
        return f"POST_G{C.NUM_GROUPS}" if self.index == C.NUM_GROUPS else f"G{self.index}_G{self.index + 1}"


@dataclass(frozen=True)
class Boundary:
    """An inert insertion point in the residual stream between two groups.

    ``before_layer`` is the layer the boundary sits in front of (``None`` for the
    final boundary); ``after_layer`` is the layer it sits behind (``None`` for the
    first). Hooks attach as a forward-pre-hook on ``before_layer`` when it exists,
    otherwise as a forward-hook on ``after_layer``.
    """

    name: str
    position: int  # 0..16
    before_layer: int | None
    after_layer: int | None
    #: 1-based groups on either side; ``None`` outside the stack.
    upstream_group: int | None
    downstream_group: int | None

    @property
    def is_pre_stack(self) -> bool:
        return self.after_layer is None

    @property
    def is_post_stack(self) -> bool:
        return self.before_layer is None


class HybridGroupView:
    """Read-only view of the 64-layer stack as 16 hybrid groups.

    Construct from the checkpoint's ``layer_types`` (the effective source of
    truth per audit 02) and validate against audited constants. The view never
    holds a reference to weights; it can be built with no model loaded.
    """

    def __init__(self, layer_types: Sequence[str]) -> None:
        self._layer_types: tuple[str, ...] = tuple(layer_types)
        self._validate_layer_types()
        self._layers: tuple[LayerSpec, ...] = tuple(
            LayerSpec(
                index=i,
                layer_type=t,
                group_index=C.group_index_of_layer(i),
                position_in_group=i % C.GROUP_SIZE,
            )
            for i, t in enumerate(self._layer_types)
        )
        self._groups: tuple[GroupSpec, ...] = tuple(
            GroupSpec(
                index=g,
                layer_indices=C.layers_of_group(g),
                layer_types=tuple(self._layer_types[i] for i in C.layers_of_group(g)),
            )
            for g in range(1, C.NUM_GROUPS + 1)
        )
        self._boundaries: tuple[Boundary, ...] = self._build_boundaries()

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_text_config(cls, text_config: Any) -> "HybridGroupView":
        """Build from a ``Qwen3_5TextConfig`` (or anything exposing ``layer_types``)."""
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise GroupStructureError(
                "text config exposes no explicit layer_types; the audit requires the "
                "explicit 64-entry list as the effective source of truth (audit 02)"
            )
        return cls(layer_types)

    @classmethod
    def from_checkpoint_config(cls, config_json: dict) -> "HybridGroupView":
        """Build from a raw ``config.json`` mapping."""
        text = config_json.get("text_config", config_json)
        return cls(text["layer_types"])

    # -- validation --------------------------------------------------------

    def _validate_layer_types(self) -> None:
        if len(self._layer_types) != C.NUM_LAYERS:
            raise GroupStructureError(
                f"expected {C.NUM_LAYERS} layer_types, got {len(self._layer_types)}"
            )
        expected = C.expected_layer_types()
        if self._layer_types != expected:
            mismatches = [
                (i, got, want)
                for i, (got, want) in enumerate(zip(self._layer_types, expected))
                if got != want
            ]
            raise GroupStructureError(
                "layer_types do not match the audited pattern "
                f"16 x {C.GROUP_PATTERN}; first mismatches: {mismatches[:5]}"
            )
        attention = tuple(
            i for i, t in enumerate(self._layer_types) if t == C.FULL_ATTENTION_TYPE
        )
        if attention != C.ATTENTION_LAYER_INDICES:
            raise GroupStructureError(
                f"full-attention indices {attention} != audited {C.ATTENTION_LAYER_INDICES}"
            )

    def validate_against_model(self, model: Any) -> dict[str, Any]:
        """Check the instantiated HF modules against this view.

        Verifies layer count, per-layer ``layer_type``, and that the token mixer
        modules are the stock ``Qwen3_5GatedDeltaNet`` / ``Qwen3_5Attention``
        classes at the expected indices (A11: no re-implemented cells).
        """
        layers = get_decoder_layers(model)
        if len(layers) != C.NUM_LAYERS:
            raise GroupStructureError(f"model has {len(layers)} layers, expected {C.NUM_LAYERS}")

        seen_classes: dict[str, str] = {}
        for spec in self._layers:
            layer = layers[spec.index]
            declared = getattr(layer, "layer_type", None)
            if declared != spec.layer_type:
                raise GroupStructureError(
                    f"layer {spec.index}: module layer_type={declared!r}, view says {spec.layer_type!r}"
                )
            if not hasattr(layer, spec.mixer_attr):
                raise GroupStructureError(
                    f"layer {spec.index}: expected mixer attribute {spec.mixer_attr!r}"
                )
            wrong_attr = "linear_attn" if spec.is_attention else "self_attn"
            if hasattr(layer, wrong_attr):
                raise GroupStructureError(
                    f"layer {spec.index}: unexpected mixer attribute {wrong_attr!r}"
                )
            mixer_cls = type(getattr(layer, spec.mixer_attr)).__name__
            seen_classes[spec.mixer_attr] = mixer_cls

        expected_classes = {
            "linear_attn": "Qwen3_5GatedDeltaNet",
            "self_attn": "Qwen3_5Attention",
        }
        for attr, cls_name in expected_classes.items():
            if seen_classes.get(attr) != cls_name:
                raise GroupStructureError(
                    f"mixer {attr!r} is {seen_classes.get(attr)!r}, expected the stock "
                    f"{cls_name!r} (A11 forbids re-implemented or copy-modified cells)"
                )
        return {
            "num_layers": len(layers),
            "mixer_classes": seen_classes,
            "attention_layer_indices": list(C.ATTENTION_LAYER_INDICES),
        }

    # -- accessors ---------------------------------------------------------

    @property
    def layer_types(self) -> tuple[str, ...]:
        return self._layer_types

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        return self._layers

    @property
    def groups(self) -> tuple[GroupSpec, ...]:
        return self._groups

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    def group(self, index: int) -> GroupSpec:
        if not 1 <= index <= C.NUM_GROUPS:
            raise ValueError(f"group index out of range: {index}")
        return self._groups[index - 1]

    def layer(self, index: int) -> LayerSpec:
        if not 0 <= index < C.NUM_LAYERS:
            raise ValueError(f"layer index out of range: {index}")
        return self._layers[index]

    def boundary(self, name: str) -> Boundary:
        for boundary in self._boundaries:
            if boundary.name == name:
                return boundary
        raise ValueError(f"unknown boundary: {name!r}; known: {BOUNDARY_NAMES}")

    def attention_layer_indices(self) -> tuple[int, ...]:
        return tuple(spec.index for spec in self._layers if spec.is_attention)

    def gdn_layer_indices(self) -> tuple[int, ...]:
        return tuple(spec.index for spec in self._layers if spec.is_gdn)

    def prefix_layers(self, through_group: int) -> tuple[int, ...]:
        """Layer indices of the contiguous prefix G1..``through_group``.

        Used later by progressive depth (part 2). In part 1 it only documents
        that every CAPE-R route is a contiguous prefix ending at a group border.
        """
        if not 1 <= through_group <= C.NUM_GROUPS:
            raise ValueError(f"group index out of range: {through_group}")
        return tuple(range(0, through_group * C.GROUP_SIZE))

    def describe(self) -> dict[str, Any]:
        return {
            "num_layers": C.NUM_LAYERS,
            "num_groups": C.NUM_GROUPS,
            "group_pattern": list(C.GROUP_PATTERN),
            "attention_layer_indices": list(self.attention_layer_indices()),
            "num_gdn_layers": len(self.gdn_layer_indices()),
            "num_attention_layers": len(self.attention_layer_indices()),
            "seq_length_anchor_layer": C.SEQ_LENGTH_ANCHOR_LAYER,
            "boundaries": [b.name for b in self._boundaries],
            "groups": [
                {
                    "index": g.index,
                    "layers": list(g.layer_indices),
                    "types": list(g.layer_types),
                    "attention_layer": g.attention_layer,
                }
                for g in self._groups
            ],
        }

    # -- internals ---------------------------------------------------------

    def _build_boundaries(self) -> tuple[Boundary, ...]:
        boundaries: list[Boundary] = []
        for position, name in enumerate(BOUNDARY_NAMES):
            upstream = position if position >= 1 else None
            downstream = position + 1 if position < C.NUM_GROUPS else None
            before_layer = self._groups[downstream - 1].first_layer if downstream is not None else None
            after_layer = self._groups[upstream - 1].last_layer if upstream is not None else None
            boundaries.append(
                Boundary(
                    name=name,
                    position=position,
                    before_layer=before_layer,
                    after_layer=after_layer,
                    upstream_group=upstream,
                    downstream_group=downstream,
                )
            )
        if len(boundaries) != C.NUM_BOUNDARIES:
            raise GroupStructureError(
                f"built {len(boundaries)} boundaries, expected {C.NUM_BOUNDARIES}"
            )
        return tuple(boundaries)


def get_decoder_layers(model: Any) -> Sequence[Any]:
    """Return the ``nn.ModuleList`` of decoder layers for either backbone mode.

    Supports ``Qwen3_5ForCausalLM`` (``model.model.layers``) and
    ``Qwen3_5ForConditionalGeneration`` (``model.model.language_model.layers``)
    without touching either implementation.
    """
    candidates: Iterable[tuple[str, Any]] = (
        ("model.model.language_model.layers", _dig(model, "model", "language_model", "layers")),
        ("model.model.layers", _dig(model, "model", "layers")),
        ("model.layers", _dig(model, "layers")),
    )
    for _, value in candidates:
        if value is not None:
            return value
    raise GroupStructureError("could not locate the decoder layer list on the given model")


def get_text_model(model: Any) -> Any:
    """Return the ``Qwen3_5TextModel`` instance for either backbone mode."""
    for path in (("model", "language_model"), ("model",)):
        candidate = _dig(model, *path)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise GroupStructureError("could not locate the text model on the given model")


def _dig(obj: Any, *attrs: str) -> Any:
    current = obj
    for attr in attrs:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current
