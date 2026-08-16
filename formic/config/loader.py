"""Strict YAML loading for :class:`~formic.config.schema.RunConfig`.

Unknown keys are fatal. A typo that silently disables a flag would make an
experiment uninterpretable, which is exactly what the plan's reproducibility
rules forbid.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from formic.config.schema import ConfigError, RunConfig

__all__ = ["load_config", "load_config_dict", "dump_config", "config_to_yaml"]


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return load_config_dict(raw)


def load_config_dict(raw: dict[str, Any]) -> RunConfig:
    """Build and validate a :class:`RunConfig` from a plain mapping."""
    config = _build(RunConfig, raw, path="")
    config.validate()
    return config


def _build(cls: type, raw: Any, path: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(
            f"{path or 'config'}: unknown key(s) {unknown}; allowed: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in raw:
            continue
        value = raw[name]
        child_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(f.type, value, child_path, cls, name)
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, path: str, owner: type, field_name: str) -> Any:
    declared = _resolve_annotation(annotation, owner)

    # Nested dataclass section.
    if is_dataclass(declared) and isinstance(value, dict):
        return _build(declared, value, path)

    default = _default_of(owner, field_name)

    # Tuple-typed fields accept YAML lists.
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)

    # dict[str, str] fields (max_memory): normalise keys to str for stable hashing.
    if isinstance(default, dict) and isinstance(value, dict):
        return {str(k): v for k, v in value.items()}

    if isinstance(default, bool) and not isinstance(value, bool):
        raise ConfigError(f"{path}: expected a boolean, got {value!r}")

    if isinstance(default, int) and not isinstance(default, bool) and isinstance(value, bool):
        raise ConfigError(f"{path}: expected a number, got a boolean")

    return value


def _resolve_annotation(annotation: Any, owner: type) -> Any:
    """Resolve a possibly-stringified annotation to a class when it is a dataclass."""
    if isinstance(annotation, str):
        import formic.config.schema as schema_module

        candidate = getattr(schema_module, annotation.split("|")[0].strip(), None)
        if candidate is not None and is_dataclass(candidate):
            return candidate
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [a for a in get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def _default_of(owner: type, field_name: str) -> Any:
    for f in fields(owner):
        if f.name != field_name:
            continue
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:  # type: ignore[misc]
            return f.default_factory()  # type: ignore[misc]
    return None


def config_to_yaml(config: RunConfig) -> str:
    """Render a config back to YAML (used to pin the exact config of a run)."""
    return yaml.safe_dump(_plain(config.to_dict()), sort_keys=True, default_flow_style=False)


def dump_config(config: RunConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_yaml(config), encoding="utf-8")
    return path


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
