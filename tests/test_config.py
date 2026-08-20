"""Run configuration: strict schema, flags OFF by default, frozen policies."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from formic.backbone import constants as C
from formic.backbone.groups import BOUNDARY_NAMES
from formic.config.loader import load_config, load_config_dict
from formic.config.schema import ConfigError, RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_config_file_loads_and_validates():
    config = load_config(CONFIGS / "default.yaml")
    config.validate()
    assert config.backbone.mode == "text_only"
    assert config.backbone.attn_implementation == "eager"
    assert config.backbone.dtype == "bfloat16"
    assert config.numerics.cublas_workspace_config == ":4096:8"
    assert config.numerics.warmup_traces_per_shape == 6
    assert config.numerics.measured_traces_per_shape == 2


def test_default_config_is_identity_mode():
    """The core invariant: default config == stock Qwen3.8-27B."""
    config = load_config(CONFIGS / "default.yaml")
    assert config.flags.all_off
    assert config.flags.any_enabled() == ()
    assert config.boundaries.enabled_observers == ()
    assert config.boundaries.enabled_insertions == ()
    assert config.identity_mode() is True


def test_every_flag_defaults_to_off():
    flags = RunConfig().flags
    assert flags.all_off
    assert flags.any_enabled() == ()


def test_part2_flags_are_refused_in_part1():
    for name in RunConfig().flags.PART2_ONLY:
        with pytest.raises(ConfigError):
            load_config_dict({"flags": {name: True}})


def test_part1_flags_are_allowed_but_break_identity_mode():
    config = load_config_dict({"flags": {"transaction_engine": True}})
    assert config.flags.transaction_engine is True
    assert config.identity_mode() is False
    assert config.flags.any_enabled() == ("transaction_engine",)


def test_unknown_keys_are_fatal():
    with pytest.raises(ConfigError):
        load_config_dict({"unknown_section": {}})
    with pytest.raises(ConfigError):
        load_config_dict({"thinking": {"mode": "on", "typo_key": 1}})


def test_control_sampling_is_frozen_to_greedy():
    with pytest.raises(ConfigError):
        load_config_dict({"sampling": {"control": "sampled"}})


def test_payload_sampling_defaults_are_the_checkpoint_defaults():
    payload = RunConfig().sampling.payload
    assert payload.temperature == C.CHECKPOINT_DEFAULT_TEMPERATURE == 1.0
    assert payload.top_p == C.CHECKPOINT_DEFAULT_TOP_P == 0.95
    assert payload.top_k == C.CHECKPOINT_DEFAULT_TOP_K == 20


def test_thinking_policy_modes():
    assert RunConfig().thinking.mode == "capped"
    assert RunConfig().thinking.cap_tokens == 4096
    assert RunConfig().thinking.enable_thinking is True
    off = load_config_dict({"thinking": {"mode": "off"}})
    assert off.thinking.enable_thinking is False
    with pytest.raises(ConfigError):
        load_config_dict({"thinking": {"mode": "sometimes"}})
    with pytest.raises(ConfigError):
        load_config_dict({"thinking": {"mode": "capped", "cap_tokens": 0}})


def test_strict_inventory_cannot_be_disabled():
    with pytest.raises(ConfigError):
        load_config_dict({"backbone": {"strict_inventory": False}})


def test_non_bf16_dtype_is_refused():
    with pytest.raises(ConfigError):
        load_config_dict({"backbone": {"dtype": "float16"}})


def test_multimodal_mode_is_refused_in_spec_01():
    with pytest.raises(ConfigError):
        load_config_dict({"backbone": {"mode": "reference_multimodal"}})


def test_boundaries_reject_unknown_names():
    with pytest.raises(ConfigError):
        load_config_dict({"boundaries": {"enabled_observers": ["G99_G100"]}})
    config = load_config_dict({"boundaries": {"enabled_observers": ["G4_G5"]}})
    assert config.boundaries.enabled_observers == ("G4_G5",)
    assert config.identity_mode() is False


def test_boundaries_reject_duplicate_names():
    with pytest.raises(ConfigError, match="duplicate"):
        load_config_dict(
            {"boundaries": {"enabled_insertions": ["PRE_G1", "PRE_G1"]}}
        )


def test_noop_hook_config_selects_all_seventeen_boundaries():
    config = load_config(CONFIGS / "step1_noop_hooks.yaml")
    assert config.flags.all_off
    assert config.boundaries.enabled_observers == ()
    assert len(config.boundaries.enabled_insertions) == C.NUM_BOUNDARIES == 17
    assert set(config.boundaries.enabled_insertions) == set(BOUNDARY_NAMES)
    assert config.identity_mode() is False


def test_config_hash_is_stable_and_sensitive():
    base = RunConfig()
    assert base.config_hash() == RunConfig().config_hash()
    changed = load_config_dict({"run": {"seed": 7}})
    assert changed.config_hash() != base.config_hash()
    assert len(base.config_hash()) == 64


def test_config_roundtrips_through_yaml():
    config = load_config(CONFIGS / "default.yaml")
    from formic.config.loader import config_to_yaml

    reloaded = load_config_dict(yaml.safe_load(config_to_yaml(config)))
    assert reloaded.config_hash() == config.config_hash()


def test_decode_stability_policy_is_strict():
    with pytest.raises(ConfigError, match="measured_traces"):
        load_config_dict({"numerics": {"measured_traces_per_shape": 1}})
    with pytest.raises(ConfigError, match="cannot be disabled"):
        load_config_dict({"numerics": {"require_last_two_exact": False}})
    with pytest.raises(ConfigError, match="cublas_workspace_config"):
        load_config_dict({"numerics": {"cublas_workspace_config": "invalid"}})


def test_reference_prompt_set_is_wellformed():
    data = yaml.safe_load((CONFIGS / "reference_prompts.yaml").read_text(encoding="utf-8"))
    assert data["version"] == 1
    ids = [p["id"] for p in data["prompts"]]
    assert len(ids) == len(set(ids))
    for prompt in data["prompts"]:
        assert prompt["kind"] in {"raw", "chat"}
        assert ("text" in prompt) ^ ("messages" in prompt)
