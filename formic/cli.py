"""Formic command line.

Step-1 commands. Weight-free commands (``verify``, ``structure``, ``inventory``,
``config``, ``env``) run in seconds and are what CI uses; ``load`` and
``generate`` touch the 55 GB checkpoint.

    python -m formic.cli verify
    python -m formic.cli structure
    python -m formic.cli inventory --json
    python -m formic.cli config --config configs/default.yaml
    python -m formic.cli env
    python -m formic.cli load --config configs/default.yaml
    python -m formic.cli generate --config configs/default.yaml --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from formic.backbone import constants as C
from formic.config.loader import config_to_yaml, load_config
from formic.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _load(config_path: str | None) -> RunConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    return load_config(path)


def cmd_config(args: argparse.Namespace) -> int:
    config = _load(args.config)
    if args.json:
        print(json.dumps({"config": config.to_dict(), "hash": config.config_hash()}, indent=2))
    else:
        print(config_to_yaml(config))
        print(f"# config_hash: {config.config_hash()}")
        print(f"# identity_mode: {config.identity_mode()}")
        print(f"# enabled_flags: {config.flags.any_enabled() or '(none)'}")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    from formic.science.determinism import environment_report

    print(json.dumps(environment_report(), indent=2))
    return 0


def cmd_structure(args: argparse.Namespace) -> int:
    from formic.backbone.groups import HybridGroupView

    config = _load(args.config)
    raw = json.loads((Path(config.backbone.checkpoint_path) / "config.json").read_text())
    view = HybridGroupView.from_checkpoint_config(raw)
    if args.json:
        print(json.dumps(view.describe(), indent=2))
        return 0

    print(f"HYBRID GROUP VIEW  ({C.NUM_LAYERS} layers / {C.NUM_GROUPS} groups)")
    print(f"  pattern           : {' + '.join(C.GROUP_PATTERN)}")
    print(f"  attention layers  : {list(view.attention_layer_indices())}")
    print(f"  gdn layers        : {len(view.gdn_layer_indices())}")
    print(f"  seq-length anchor : layer {C.SEQ_LENGTH_ANCHOR_LAYER} (first full attention)")
    print()
    print("  group | layers        | types                       | boundaries")
    print("  ------+---------------+-----------------------------+---------------------")
    for group in view.groups:
        types = " ".join("A" if t == C.FULL_ATTENTION_TYPE else "G" for t in group.layer_types)
        layers = f"{group.first_layer:>2}-{group.last_layer:<2}"
        print(
            f"  G{group.index:<4} | {layers:<13} | {types:<27} | "
            f"{group.entry_boundary} -> {group.exit_boundary}"
        )
    print()
    print(f"  boundaries ({len(view.boundaries)}): " + ", ".join(b.name for b in view.boundaries))
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    from formic.backbone.inventory import CheckpointInventory

    config = _load(args.config)
    inventory = CheckpointInventory.from_checkpoint(config.backbone.checkpoint_path)
    inventory.validate_against_audit()
    summary = inventory.summary()
    if args.json:
        summary["expected_text_only_tensors"] = len(inventory.expected_model_tensors("text_only"))
        summary["declared_exclusions_text_only"] = inventory.declared_exclusions("text_only")
        print(json.dumps(summary, indent=2))
        return 0

    print("CHECKPOINT INVENTORY (strict, headers only - no weights read)")
    print(f"  path          : {summary['checkpoint_path']}")
    print(f"  tensors       : {summary['num_tensors']:,} in {summary['num_shards']} shards")
    print(f"  stored params : {summary['total_params']:,}")
    print(f"  payload bytes : {summary['total_bytes']:,}")
    print(f"  dtypes        : {summary['dtypes']}")
    for family, stats in summary["families"].items():
        print(f"    {family:<9}: {stats['tensors']:>4} tensors / {stats['params']:>15,} params")
    print()
    for mode in ("text_only", "audit_multimodal"):
        expected = inventory.expected_model_tensors(mode)  # type: ignore[arg-type]
        params = sum(_numel(shape) for shape in expected.values())
        print(
            f"  mode {mode:<21}: {len(expected):>4} tensors / {params:>15,} params "
            f"| excluded {inventory.declared_exclusions(mode)}"  # type: ignore[arg-type]
        )
    print("\n  AUDIT CONFORMANCE: PASS")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Weight-free structural verification (CI entry point)."""
    from formic.backbone.loader import verify_checkpoint_only

    config = _load(args.config)
    checks: list[tuple[str, bool, str]] = []

    try:
        result = verify_checkpoint_only(config.backbone.checkpoint_path)
        checks.append(("checkpoint inventory vs audit", True, ""))
        checks.append(("hybrid group structure", True, ""))
    except Exception as exc:  # noqa: BLE001 - report and fail
        checks.append(("checkpoint inventory / structure", False, str(exc)))
        result = {}

    try:
        config.validate()
        checks.append(("config schema", True, ""))
    except Exception as exc:  # noqa: BLE001
        checks.append(("config schema", False, str(exc)))

    identity = config.identity_mode()
    checks.append(
        (
            "identity mode (all flags OFF, no boundary hooks)",
            identity,
            "" if identity else f"enabled: {config.flags.any_enabled()}",
        )
    )

    print("FORMIC VERIFY (no weights loaded)")
    for name, ok, detail in checks:
        print(f"  {name:<50} {'PASS' if ok else 'FAIL'}")
        if detail:
            print(f"      {detail}")
    if result:
        print(f"\n  expected params  text_only : {result['expected_text_only_params']:,}")
        print(f"  audited params   multimodal: {result['expected_multimodal_params']:,}")
    ok = all(ok for _, ok, _ in checks)
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_load(args: argparse.Namespace) -> int:
    from formic.backbone.loader import load_backbone

    config = _load(args.config)
    handle = load_backbone(config)
    print(json.dumps(handle.describe(), indent=2, default=str))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from formic.backbone.loader import load_backbone
    from formic.backbone.runner import generate, render_chat_prompt

    config = _load(args.config)
    handle = load_backbone(config)
    if args.chat:
        prompt = render_chat_prompt(handle, [{"role": "user", "content": args.prompt}])
    else:
        prompt = args.prompt
    result = generate(
        handle,
        prompt,
        do_sample=args.sample,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"--- prompt ({len(result.prompt_token_ids)} tokens) ---")
        print(prompt)
        print(f"--- output ({len(result.generated_token_ids)} tokens, {result.seconds:.1f}s) ---")
        print(result.text)
    return 0


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formic", description="Formic CLI (part 1)")
    parser.add_argument("--config", default=None, help="path to a run config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="show the resolved run config and its hash")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("env", help="environment/backend report")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("structure", help="hybrid group view (no weights)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_structure)

    p = sub.add_parser("inventory", help="strict checkpoint inventory (no weights)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("verify", help="weight-free verification, CI entry point")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("load", help="load the backbone and print the strict-load report")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("generate", help="generate from a prompt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true", help="render through the native chat template")
    p.add_argument("--sample", action="store_true", help="payload sampling instead of greedy")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
