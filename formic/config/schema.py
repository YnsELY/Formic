"""Run configuration schema.

Plan rule: *toute exécution est entièrement décrite par sa config*. A run is
reproducible from (config hash, git commit, seeds) alone, so the schema is
strict: unknown keys are a hard error, and every new behaviour has a flag that
defaults to OFF (plan rule 3).

Frozen cross-cutting policies encoded here:

* 2.1 thinking  - native ``<think>`` segment allowed before any typed action,
  hard cap, mode pinned in every run.
* 2.2 sampling  - control fields are greedy ALWAYS; payload uses checkpoint
  defaults until the step-3 temperature sweep decides otherwise.
* 2.4 numerics  - kernels/backends are part of the config because bit-exactness
  is only required at identical config and backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal

from formic.backbone import constants as C

CONFIG_VERSION = 1

ThinkingMode = Literal["on", "off", "capped"]
BackboneMode = Literal["text_only"]


class ConfigError(ValueError):
    """Raised on any schema violation. Never softened: a bad config is fatal."""


# --------------------------------------------------------------------------
# Leaf sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSection:
    name: str = "unnamed"
    #: Free-text experiment id (``EXP-0001``) linking this run to the registry.
    experiment_id: str | None = None
    seed: int = 0
    #: Extra seeds for multi-seed measurements (>=3 required for decisive runs).
    seeds: tuple[int, ...] = (0,)
    deterministic: bool = True
    notes: str = ""

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("run.seed must be >= 0")
        if not self.seeds:
            raise ConfigError("run.seeds must not be empty")
        if any(s < 0 for s in self.seeds):
            raise ConfigError("run.seeds entries must be >= 0")


@dataclass(frozen=True)
class BackboneSection:
    checkpoint_path: str = "/workspace/Qwen3.8-27B"
    #: The only part-1 runtime mode: Qwen3_5ForCausalLM, with no vision tower (A7).
    mode: BackboneMode = "text_only"
    dtype: str = "bfloat16"
    #: ``eager`` makes masks explicit and is the audited backend; keep it as the
    #: default so identity comparisons stay meaningful (audit 11).
    attn_implementation: str = "eager"
    device_map: str | None = "auto"
    max_memory: dict[str, str] = field(default_factory=lambda: {"0": "40GiB", "cpu": "300GiB"})
    #: A12: strict tensor inventory. Permissive loading must be impossible.
    strict_inventory: bool = True
    #: Fail if the loaded module tree contains a vision tower in text_only mode.
    assert_no_vision_tower: bool = True

    def validate(self) -> None:
        if self.mode != "text_only":
            raise ConfigError(
                "backbone.mode must be 'text_only' in part 1; multimodal execution "
                "is outside SPEC-01"
            )
        if self.dtype != "bfloat16":
            raise ConfigError(
                "backbone.dtype must be bfloat16: BF16 is the reference for every "
                "decisive measurement (plan rule 6)"
            )
        if not self.strict_inventory:
            raise ConfigError("backbone.strict_inventory cannot be disabled (audit constraint A12)")


@dataclass(frozen=True)
class ThinkingSection:
    """Plan 2.1. The checkpoint is thinking-default; the segment stays available."""

    mode: ThinkingMode = "capped"
    cap_tokens: int = 4096
    #: Scratch content is non-authoritative: logged for audit, never parsed as an
    #: action, never used to satisfy a criterion, never persisted as state.
    log_scratch: bool = True

    def validate(self) -> None:
        if self.mode not in ("on", "off", "capped"):
            raise ConfigError(f"thinking.mode invalid: {self.mode}")
        if self.mode == "capped" and self.cap_tokens <= 0:
            raise ConfigError("thinking.cap_tokens must be > 0 in capped mode")

    @property
    def enable_thinking(self) -> bool:
        """Value passed to the chat template's ``enable_thinking`` switch."""
        return self.mode != "off"


@dataclass(frozen=True)
class PayloadSampling:
    do_sample: bool = True
    temperature: float = C.CHECKPOINT_DEFAULT_TEMPERATURE
    top_p: float = C.CHECKPOINT_DEFAULT_TOP_P
    top_k: int = C.CHECKPOINT_DEFAULT_TOP_K

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigError("sampling.payload.temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ConfigError("sampling.payload.top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ConfigError("sampling.payload.top_k must be >= 0")


@dataclass(frozen=True)
class SamplingSection:
    """Plan 2.2. ``control`` is frozen to greedy and cannot be overridden."""

    control: Literal["greedy"] = "greedy"
    payload: PayloadSampling = field(default_factory=PayloadSampling)
    #: Scratch uses payload settings, inside the thinking cap.
    scratch_follows_payload: bool = True

    def validate(self) -> None:
        if self.control != "greedy":
            raise ConfigError(
                "sampling.control is frozen to 'greedy' by plan 2.2 (control fields: "
                "action type, target IDs, paths, hashes, statuses, state transitions)"
            )
        self.payload.validate()


@dataclass(frozen=True)
class BoundariesSection:
    """The 17 group-boundary insertion points, all inert by default.

    ``enabled_observers`` attaches read-only hooks. ``enabled_insertions`` may
    select no-op hooks for SPEC-01's inertness proof; transforming callbacks
    remain reserved for later steps. With both lists empty, *no hook is
    registered at all*, so the forward pass is the stock HF graph.
    """

    enabled_observers: tuple[str, ...] = ()
    enabled_insertions: tuple[str, ...] = ()

    def validate(self) -> None:
        from formic.backbone.groups import BOUNDARY_NAMES

        if len(set(self.enabled_observers)) != len(self.enabled_observers):
            raise ConfigError("boundaries.enabled_observers contains duplicate names")
        if len(set(self.enabled_insertions)) != len(self.enabled_insertions):
            raise ConfigError("boundaries.enabled_insertions contains duplicate names")
        for name in self.enabled_observers:
            if name not in BOUNDARY_NAMES:
                raise ConfigError(f"unknown boundary in enabled_observers: {name}")
        for name in self.enabled_insertions:
            if name not in BOUNDARY_NAMES:
                raise ConfigError(f"unknown boundary in enabled_insertions: {name}")


@dataclass(frozen=True)
class FlagsSection:
    """Every future Formic mechanism, OFF by default (plan rule 3).

    The exhaustive list is intentional: it makes the roadmap auditable and makes
    "all flags OFF == Qwen3.8-27B" a checkable property rather than a promise.
    Part-1 steps may only flip flags they own.
    """

    # step 2
    snapshot_restore: bool = False
    # step 4-6
    transaction_engine: bool = False
    contract_compiler: bool = False
    state_fabric: bool = False
    reference_monitor: bool = False
    typed_actions: bool = False
    # step 8
    role_trust_embeddings: bool = False
    control_token_rows: bool = False
    decision_slots: bool = False
    action_head: bool = False
    pointer_heads: bool = False
    completion_head: bool = False
    stop_uncertainty_head: bool = False
    instruction_echo_probe: bool = False
    # part 2 (must stay False for the whole of part 1)
    l1_exit_bridge: bool = False
    l0_exit_bridge: bool = False
    route_conditional_lora: bool = False
    anytime_exit_gating: bool = False
    continue_act_head: bool = False
    hspc_prefix_cache: bool = False
    dspd_speculation: bool = False
    gdn_rollback_protocol: bool = False
    mtp: bool = False
    constrained_decoding_fsm: bool = False

    #: Flags that part 1 must never enable (checked by :meth:`validate`).
    PART2_ONLY = (
        "l1_exit_bridge",
        "l0_exit_bridge",
        "route_conditional_lora",
        "anytime_exit_gating",
        "continue_act_head",
        "hspc_prefix_cache",
        "dspd_speculation",
        "gdn_rollback_protocol",
        "mtp",
        "constrained_decoding_fsm",
    )

    def validate(self) -> None:
        for name in self.PART2_ONLY:
            if getattr(self, name):
                raise ConfigError(
                    f"flags.{name} belongs to part 2 and must stay OFF until FORMIC-M1 "
                    "is validated (plan: 'Ce qui est volontairement repoussé')"
                )

    def any_enabled(self) -> tuple[str, ...]:
        return tuple(
            f.name for f in fields(self) if isinstance(getattr(self, f.name), bool) and getattr(self, f.name)
        )

    @property
    def all_off(self) -> bool:
        return not self.any_enabled()


@dataclass(frozen=True)
class GenerationSection:
    """Generation limits for a single call (not a behavioural flag)."""

    max_new_tokens: int = 128
    #: ``None`` -> use the checkpoint's generation config EOS ids.
    eos_token_ids: tuple[int, ...] = C.GENERATION_EOS_TOKEN_IDS

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigError("generation.max_new_tokens must be > 0")


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    formic_config_version: int = CONFIG_VERSION
    run: RunSection = field(default_factory=RunSection)
    backbone: BackboneSection = field(default_factory=BackboneSection)
    thinking: ThinkingSection = field(default_factory=ThinkingSection)
    sampling: SamplingSection = field(default_factory=SamplingSection)
    boundaries: BoundariesSection = field(default_factory=BoundariesSection)
    flags: FlagsSection = field(default_factory=FlagsSection)
    generation: GenerationSection = field(default_factory=GenerationSection)

    def validate(self) -> None:
        if self.formic_config_version != CONFIG_VERSION:
            raise ConfigError(
                f"formic_config_version {self.formic_config_version} != {CONFIG_VERSION}"
            )
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and hasattr(value, "validate"):
                value.validate()

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        """Stable serialisation used for the config hash recorded in every run."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=_json_default)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def identity_mode(self) -> bool:
        """True when this config must reproduce stock Qwen3.8-27B exactly."""
        return (
            self.flags.all_off
            and not self.boundaries.enabled_observers
            and not self.boundaries.enabled_insertions
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
