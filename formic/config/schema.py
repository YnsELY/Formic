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
PromptLengthClass = Literal["short", "medium", "long"]


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


@dataclass(frozen=True)
class NumericsSection:
    """Pinned CUDA backend and cached-decode warmup policy (plan 2.4)."""

    cublas_workspace_config: str = ":4096:8"
    cudnn_allow_tf32: bool = False
    cuda_matmul_allow_tf32: bool = False
    flash_sdp: bool = False
    mem_efficient_sdp: bool = False
    math_sdp: bool = True
    warmup_traces_per_shape: int = 6
    measured_traces_per_shape: int = 2
    require_last_two_exact: bool = True

    def validate(self) -> None:
        if self.cublas_workspace_config not in (":16:8", ":4096:8"):
            raise ConfigError(
                "numerics.cublas_workspace_config must be ':16:8' or ':4096:8'"
            )
        if self.warmup_traces_per_shape < 0:
            raise ConfigError("numerics.warmup_traces_per_shape must be >= 0")
        if self.measured_traces_per_shape < 2:
            raise ConfigError(
                "numerics.measured_traces_per_shape must be >= 2 for stability checking"
            )
        if not self.require_last_two_exact:
            raise ConfigError("numerics.require_last_two_exact cannot be disabled")


@dataclass(frozen=True)
class IdentitySection:
    """SPEC-02 identity and calibration protocol.

    These are measurement controls, not Formic runtime mechanisms. Keeping
    them in the resolved config makes every verdict reproducible without
    weakening the all-flags-off identity invariant.
    """

    prompt_set_path: str = "configs/reference_prompts.yaml"
    tolerances_path: str = "tolerances.json"
    tolerance_governance_path: str = "configs/tolerance_governance.json"
    verdict_path: str = "reports/identity/latest_verdict.json"
    backbone_hash_path: str = (
        "configs/checkpoint_metadata/qwen3_8_27b/backbone_hash.json"
    )
    decode_tokens: int = 8
    accumulation_probe_tokens: int = 64
    measurement_repetitions: int = 3
    exact_gate_repetitions: int = 2
    continuation_seeds: tuple[int, ...] = (0, 1, 2)
    decode_prompt_ids: tuple[str, ...] = (
        "short_error_assertion",
        "medium_cache_regression",
        "long_resume_incidents",
    )
    accumulation_probe_prompt_ids: tuple[str, ...] = (
        "short_error_assertion",
        "medium_cache_regression",
    )
    snapshot_validation_prompt_id: str = "audit_echo"
    tolerance_margin_multiplier: float = 2.0
    ci_segmentations: tuple[str, ...] = ("median",)
    calibration_segmentations: tuple[str, ...] = (
        "early",
        "median",
        "late",
        "quarters",
    )
    long_calibration_segmentations: tuple[str, ...] = ("median", "quarters")
    recompute_classes: tuple[PromptLengthClass, ...] = ("short", "medium")
    full_boundary_capture_classes: tuple[PromptLengthClass, ...] = (
        "short",
        "medium",
    )
    final_state_only_classes: tuple[PromptLengthClass, ...] = ("long",)
    require_top1_agreement: bool = True
    kl_is_blocking: bool = False

    def validate(self) -> None:
        if self.decode_tokens != 8:
            raise ConfigError(
                "identity.decode_tokens is pinned to 8 for calibration and CI by ADR-0005"
            )
        if self.accumulation_probe_tokens != 64:
            raise ConfigError(
                "identity.accumulation_probe_tokens is pinned to 64 by SPEC-02"
            )
        if self.measurement_repetitions != 3:
            raise ConfigError("identity.measurement_repetitions is pinned to 3")
        if self.exact_gate_repetitions != 2:
            raise ConfigError("identity.exact_gate_repetitions is pinned to 2")
        if len(set(self.continuation_seeds)) != 3 or any(
            seed < 0 for seed in self.continuation_seeds
        ):
            raise ConfigError(
                "identity.continuation_seeds requires 3 distinct non-negative seeds"
            )
        if self.decode_prompt_ids != (
            "short_error_assertion",
            "medium_cache_regression",
            "long_resume_incidents",
        ):
            raise ConfigError(
                "identity.decode_prompt_ids must contain one pinned prompt per class"
            )
        if self.accumulation_probe_prompt_ids != (
            "short_error_assertion",
            "medium_cache_regression",
        ):
            raise ConfigError(
                "identity.accumulation_probe_prompt_ids are pinned by ADR-0005"
            )
        if self.snapshot_validation_prompt_id != "audit_echo":
            raise ConfigError(
                "identity.snapshot_validation_prompt_id must be audit_echo"
            )
        if self.tolerance_margin_multiplier != 2.0:
            raise ConfigError(
                "identity.tolerance_margin_multiplier is fixed at 2.0 by SPEC-02"
            )
        if self.ci_segmentations != ("median",):
            raise ConfigError("identity.ci_segmentations must be exactly ['median']")
        if self.calibration_segmentations != (
            "early",
            "median",
            "late",
            "quarters",
        ):
            raise ConfigError(
                "identity.calibration_segmentations must be early/median/late/quarters"
            )
        if self.long_calibration_segmentations != ("median", "quarters"):
            raise ConfigError(
                "long prompts must use median and quarters segmentations only"
            )
        if self.recompute_classes != ("short", "medium"):
            raise ConfigError(
                "full-recomputation decode is restricted to short and medium prompts"
            )
        if set(self.full_boundary_capture_classes) != {"short", "medium"}:
            raise ConfigError(
                "full boundary capture is restricted to short and medium prompts"
            )
        if self.final_state_only_classes != ("long",):
            raise ConfigError("long prompts must use final-state-only capture")
        if not self.require_top1_agreement:
            raise ConfigError("identity.require_top1_agreement cannot be disabled")
        if self.kl_is_blocking:
            raise ConfigError("identity.kl_is_blocking must stay false")


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
    numerics: NumericsSection = field(default_factory=NumericsSection)
    identity: IdentitySection = field(default_factory=IdentitySection)

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
