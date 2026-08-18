"""Backbone loading: Qwen3.8-27B as an untouched neural substrate.

SPEC-01 loads the *stock* Hugging Face ``Qwen3_5ForCausalLM`` implementation —
no cell is re-implemented and no module is copy-modified (A11). A pure key
renaming ``model.language_model.* -> model.*`` is checked as a strict bijection
before loading. This satisfies A7 structurally: the selected class never
constructs a vision tower.

    Equivalence argument (verified in the runtime source, proved empirically by
    the step-2 identity suite): ``Qwen3_5Model.forward`` on pure text takes the
    ``position_ids = None`` branch of ``compute_3d_position_ids`` and delegates
    to the same ``Qwen3_5TextModel`` with the same arguments, and
    ``Qwen3_5Model.language_model`` *is* a ``Qwen3_5TextModel``. The two paths
    therefore run the same modules on the same inputs.

Loading is strict (A12): the checkpoint inventory is validated first, the
loaded tensor set is matched back afterwards, and every intentional
exclusion (vision in text-only, MTP always — A10) is declared and counted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from formic.backbone import constants as C
from formic.backbone.boundaries import BoundaryHookManager, count_registered_hooks
from formic.backbone.groups import HybridGroupView
from formic.backbone.inventory import (
    CheckpointInventory,
    InventoryError,
    StrictLoadReport,
    assert_strict_load,
)
from formic.config.schema import RunConfig

__all__ = ["BackboneHandle", "load_backbone", "load_tokenizer", "BackboneLoadError"]


class BackboneLoadError(RuntimeError):
    """Raised when the backbone cannot be loaded under part-1 rules."""


@dataclass
class BackboneHandle:
    """A loaded backbone plus everything needed to describe the run."""

    model: Any
    tokenizer: Any
    view: HybridGroupView
    inventory: CheckpointInventory
    load_report: StrictLoadReport
    config: RunConfig
    boundary_manager: BoundaryHookManager
    hf_loading_info: dict[str, Any] = field(default_factory=dict)
    structure_report: dict[str, Any] = field(default_factory=dict)
    load_seconds: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return self.config.backbone.mode

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model_class": type(self.model).__name__,
            "text_model_class": type(_text_model(self.model)).__name__,
            "parameters": self.parameter_count(),
            "parameter_bytes": self.parameter_count() * 2,
            "load_seconds": round(self.load_seconds, 2),
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
            "registered_layer_hooks": count_registered_hooks(self.model),
            "identity_mode": self.config.identity_mode(),
            "config_hash": self.config.config_hash(),
            "memory": self.memory,
            "structure": self.structure_report,
            "load_report": self.load_report.to_dict(),
            "hf_loading_info": self.hf_loading_info,
            "key_mapping": self.inventory.text_only_mapping_report(),
        }


def load_tokenizer(checkpoint_path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint_path))


def load_backbone(config: RunConfig, *, verbose: bool = True) -> BackboneHandle:
    """Load the backbone described by ``config`` under strict part-1 rules."""
    from formic.science.determinism import configure_determinism

    config.validate()
    configure_determinism(config.run.seed, config.run.deterministic)
    backbone_cfg = config.backbone
    path = Path(backbone_cfg.checkpoint_path)
    if not path.is_dir():
        raise BackboneLoadError(f"checkpoint directory not found: {path}")

    # --- 1. inventory before touching any weight (A12) --------------------
    inventory = CheckpointInventory.from_checkpoint(path)
    inventory.validate_against_audit()
    if verbose:
        summary = inventory.summary()
        print(
            f"[formic] inventory: {summary['num_tensors']} tensors / "
            f"{summary['total_params']:,} params across {summary['num_shards']} shards"
        )
        print(f"[formic] declared exclusions: {inventory.declared_exclusions(backbone_cfg.mode)}")

    # --- 2. structural view from the checkpoint config (A11) --------------
    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    view = HybridGroupView.from_checkpoint_config(raw_config)

    # --- 3. load stock HF modules ----------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model, hf_loading_info = _load_model(
        path, backbone_cfg, raw_config, inventory, verbose=verbose
    )
    load_seconds = time.time() - started
    model.eval()

    # --- 4. post-load verification ---------------------------------------
    report = assert_strict_load(model, inventory, backbone_cfg.mode)
    structure_report = view.validate_against_model(model)

    if backbone_cfg.assert_no_vision_tower and backbone_cfg.mode == "text_only":
        if _has_vision_tower(model):
            raise BackboneLoadError("vision tower present in text-only mode (violates A7)")

    manager = BoundaryHookManager.from_config(model, view, config.boundaries)
    manager.attach()
    hooks = count_registered_hooks(model)
    if hooks != manager.num_active_hooks:
        manager.detach()
        raise BackboneLoadError(
            f"decoder stack reports {hooks} hooks, manager attached "
            f"{manager.num_active_hooks}"
        )
    if config.identity_mode() and hooks:
        manager.detach()
        raise BackboneLoadError(
            f"{hooks} layer hook(s) registered while every flag is OFF; identity mode "
            "requires an unmodified forward graph"
        )

    attn = getattr(model.config, "_attn_implementation", None)
    if attn != backbone_cfg.attn_implementation:
        raise BackboneLoadError(
            f"attn_implementation is {attn!r}, config asked for "
            f"{backbone_cfg.attn_implementation!r}"
        )

    memory = _memory_stats(model)
    handle = BackboneHandle(
        model=model,
        tokenizer=load_tokenizer(path),
        view=view,
        inventory=inventory,
        load_report=report,
        config=config,
        boundary_manager=manager,
        hf_loading_info=hf_loading_info,
        structure_report=structure_report,
        load_seconds=load_seconds,
        memory=memory,
    )
    if verbose:
        print(f"[formic] loaded {type(model).__name__} in {load_seconds:.1f}s")
        print(report.render())
    return handle


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _load_model(
    path: Path,
    backbone_cfg: Any,
    raw_config: dict,
    inventory: CheckpointInventory,
    *,
    verbose: bool,
) -> tuple[Any, dict[str, Any]]:
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    dtype = getattr(torch, backbone_cfg.dtype)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": backbone_cfg.attn_implementation,
    }
    if backbone_cfg.device_map:
        kwargs["device_map"] = backbone_cfg.device_map
        kwargs["max_memory"] = _normalise_max_memory(backbone_cfg.max_memory)

    text_config = Qwen3_5TextConfig(**raw_config["text_config"])
    # A7: Qwen3_5ForCausalLM builds Qwen3_5TextModel only - no vision tower.
    inventory_mapping = inventory.key_mapping("text_only")
    mapping_report = inventory.text_only_mapping_report()
    if not all(
        mapping_report[key]
        for key in (
            "injective",
            "surjective_onto_expected",
            "roundtrip",
            "metadata_preserved",
            "regex_matches_name_map",
        )
    ):
        raise BackboneLoadError(f"text-only key mapping is not bijective: {mapping_report}")
    if verbose:
        print(
            f"[formic] text-only load, key_mapping={inventory_mapping}, "
            f"bijection={mapping_report['source_tensors']} tensors"
        )
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        str(path),
        config=text_config,
        key_mapping=inventory_mapping,
        output_loading_info=True,
        **kwargs,
    )
    return model, validate_hf_loading_info(loading_info, inventory)


def validate_hf_loading_info(
    loading_info: dict[str, Any], inventory: CheckpointInventory
) -> dict[str, Any]:
    """Make Transformers' source-to-target loading diagnostics fatal (A12)."""
    declared_exclusion_names = {
        record.name
        for family in ("vision", "mtp")
        for record in inventory.by_family(family)  # type: ignore[arg-type]
    }
    missing = tuple(sorted(loading_info.get("missing_keys", ())))
    reported_unexpected = tuple(sorted(loading_info.get("unexpected_keys", ())))
    unexpected = tuple(
        name for name in reported_unexpected if name not in declared_exclusion_names
    )
    mismatched = tuple(loading_info.get("mismatched_keys", ()))
    errors = tuple(loading_info.get("error_msgs", ()))
    if missing or unexpected or mismatched or errors:
        raise BackboneLoadError(
            "Transformers loading_info is not strict: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"mismatched={mismatched[:5]}, errors={errors[:5]}"
        )
    return {
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_keys": 0,
        "error_messages": 0,
        "reported_declared_exclusions": sum(
            name in declared_exclusion_names for name in reported_unexpected
        ),
    }


def _normalise_max_memory(max_memory: dict[str, str]) -> dict[Any, str]:
    """YAML keys are strings; accelerate expects int keys for GPU indices."""
    normalised: dict[Any, str] = {}
    for key, value in max_memory.items():
        normalised[int(key) if str(key).isdigit() else key] = value
    return normalised


def _has_vision_tower(model: Any) -> bool:
    inner = getattr(model, "model", None)
    if inner is None:
        return False
    if hasattr(inner, "visual"):
        return True
    return any("visual" in name.split(".") for name, _ in model.named_modules())


def _text_model(model: Any) -> Any:
    from formic.backbone.groups import get_text_model

    return get_text_model(model)


def _memory_stats(model: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }
    devices: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.device)
        devices[key] = devices.get(key, 0) + param.numel() * param.element_size()
    stats["bytes_by_device"] = devices
    if torch.cuda.is_available():
        stats["cuda_allocated"] = torch.cuda.memory_allocated()
        stats["cuda_max_allocated"] = torch.cuda.max_memory_allocated()
        stats["cuda_reserved"] = torch.cuda.memory_reserved()
    return stats


def expected_parameter_count(mode: str, inventory: CheckpointInventory) -> int:
    """Parameter count a correct load must produce, from the inventory alone."""
    expected = inventory.expected_model_tensors(mode)  # type: ignore[arg-type]
    total = 0
    for shape in expected.values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    return total


def verify_checkpoint_only(checkpoint_path: str | Path) -> dict[str, Any]:
    """Inventory + structure checks with no weights loaded (fast CI path)."""
    path = Path(checkpoint_path)
    inventory = CheckpointInventory.from_checkpoint(path)
    inventory.validate_against_audit()
    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    view = HybridGroupView.from_checkpoint_config(raw_config)
    if raw_config["architectures"] != [C.RUNTIME_ARCHITECTURE]:
        raise InventoryError(
            f"architectures {raw_config['architectures']} != ['{C.RUNTIME_ARCHITECTURE}']"
        )
    return {
        "inventory": inventory.summary(),
        "structure": view.describe(),
        "expected_text_only_params": expected_parameter_count("text_only", inventory),
        "expected_multimodal_params": expected_parameter_count("audit_multimodal", inventory),
    }
