"""Automated guard for audit constraint A11.

The 16 hybrid groups are a *view* over intact Hugging Face modules. Formic must
not re-implement, copy-modify, or monkeypatch a cell. Code review remains the
authority (step-1 checklist), but these tests turn the most common violations
into CI failures.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "formic"

#: Class names that would indicate a re-implemented cell.
FORBIDDEN_CLASS_PATTERNS = (
    r".*DecoderLayer$",
    r".*GatedDeltaNet$",
    r".*RMSNorm(Gated)?$",
    r".*RotaryEmbedding$",
    r"^Qwen3_5.*",
)

#: Cell-internal maths that has no business living in Formic code.
FORBIDDEN_SOURCE_PATTERNS = (
    (r"\bapply_rotary_pos_emb\b", "rotary application belongs to the HF cell"),
    (r"\brepeat_kv\b", "GQA expansion belongs to the HF attention cell"),
    (r"\btorch_chunk_gated_delta_rule\b", "GDN delta rule belongs to the HF cell"),
    (r"\btorch_recurrent_gated_delta_rule\b", "GDN recurrence belongs to the HF cell"),
    (r"\btorch_causal_conv1d_update\b", "GDN conv update belongs to the HF cell"),
    (r"\bsoftplus\s*\(", "GDN decay computation belongs to the HF cell"),
    (r"A_log", "GDN decay parameters must never be recomputed in Formic"),
)

#: Attribute writes onto the Hugging Face implementation (monkeypatching).
MONKEYPATCH_PATTERN = re.compile(
    r"(setattr\s*\(\s*(transformers|modeling_qwen3_5|Qwen3_5\w+))|"
    r"(modeling_qwen3_5\.\w+\s*=)|"
    r"(Qwen3_5\w+\.\w+\s*=)"
)


def _python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_package_has_python_files():
    assert _python_files(), "no Formic source files found"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_cell_class_is_redefined(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for pattern in FORBIDDEN_CLASS_PATTERNS:
            assert not re.match(pattern, node.name), (
                f"{path.name}:{node.lineno} defines {node.name!r}, which looks like a "
                "re-implemented Qwen cell (A11)"
            )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_cell_internal_maths_in_formic_code(path: Path):
    source = path.read_text(encoding="utf-8")
    code_lines = [
        line
        for line in source.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    # Docstrings quote audited formulas on purpose; strip them before scanning.
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    for pattern, reason in FORBIDDEN_SOURCE_PATTERNS:
        assert not re.search(pattern, code), f"{path.name}: {reason} (A11) [{pattern}]"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_monkeypatching_of_the_hf_implementation(path: Path):
    source = re.sub(r'"""(?:.|\n)*?"""', "", path.read_text(encoding="utf-8"))
    match = MONKEYPATCH_PATTERN.search(source)
    assert match is None, (
        f"{path.name}: looks like it patches the Hugging Face implementation "
        f"({match.group(0)!r}); Formic wraps, never rewrites (A11)"
    )


def test_no_vendored_copy_of_the_qwen_modeling_file():
    """A copied-modified modeling file is the exact failure A11 forbids."""
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        assert "class Qwen3_5DecoderLayer" not in source
        assert "def torch_chunk_gated_delta_rule" not in source


def test_backbone_uses_stock_classes_only():
    """The loader must import the official classes, not derive from them."""
    loader = (PACKAGE_ROOT / "backbone" / "loader.py").read_text(encoding="utf-8")
    assert "from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration" in loader
    tree = ast.parse(loader)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ast.unparse(base)
                assert not base_name.startswith("Qwen3_5"), (
                    f"{node.name} subclasses {base_name}; Formic must not subclass "
                    "a Qwen module in part 1 (A11)"
                )
