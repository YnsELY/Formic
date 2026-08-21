#!/usr/bin/env python3
"""Materialise the six reviewable, non-frozen SPEC-02 prompt candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from formic.backbone.constants import CHECKPOINT_COMMIT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "artifacts" / "step2" / "hf_nonweights"
OUTPUT_PATH = REPO_ROOT / "configs" / "reference_prompt_candidates.yaml"


class LiteralDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


LiteralDumper.add_representer(str, _str_representer)


def _extend_to(tokenizer, base: str, block, minimum: int, maximum: int) -> str:
    text = base.rstrip() + "\n"
    index = 1
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < minimum:
        text += block(index)
        index += 1
    count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if count > maximum:
        raise RuntimeError(f"candidate exceeded range: {count} > {maximum}")
    return text


def candidates(tokenizer):
    short_error = (
        "Fix the failing test without changing the public API:\n"
        "AssertionError: expected cache length 17, got 16."
    )
    short_diff = (
        "Review this patch and return only the corrected diff:\n"
        "- return users[index + 1]\n"
        "+ return users[index]"
    )

    medium_cache = _extend_to(
        tokenizer,
        """Investigate this Python cache regression and propose the smallest safe patch.
Do not change public signatures, do not disable validation, and preserve batch-one semantics.

```python
class SessionCache:
    def __init__(self):
        self.items = {}

    def put(self, key, value, generation):
        self.items[key] = (generation, value)

    def get(self, key, generation):
        stored_generation, value = self.items[key]
        if stored_generation <= generation:
            return value
        raise KeyError(key)
```

The failing suite is deterministic. Relevant observations follow.
""",
        lambda i: (
            f"case {i:02d}: key='worker-{i % 4}', requested_generation={i}, "
            f"stored_generation={max(0, i - 1)}, expected='miss', actual='hit'\n"
        ),
        300,
        420,
    )
    medium_diff = _extend_to(
        tokenizer,
        """Apply the requested refactor to the unified diff below. Keep behavior unchanged,
retain all error messages, and do not touch files outside src/jobs and tests/jobs.

```diff
diff --git a/src/jobs/runner.py b/src/jobs/runner.py
index 10ab120..75cd042 100644
--- a/src/jobs/runner.py
+++ b/src/jobs/runner.py
@@ -18,8 +18,10 @@ def execute(job, store):
-    result = store.load(job.id)
-    return dispatch(job, result)
+    snapshot = store.load(job.id)
+    return dispatch(job, snapshot)

diff --git a/tests/jobs/test_runner.py b/tests/jobs/test_runner.py
index 2f4d110..8a219fd 100644
--- a/tests/jobs/test_runner.py
+++ b/tests/jobs/test_runner.py
@@ -31,4 +31,5 @@ def test_execute_uses_saved_state(store):
     result = execute(job, store)
     assert result.status == "done"
```

Review notes from CI:
""",
        lambda i: (
            f"- shard {i:02d}: tests/jobs/test_runner_{i % 7}.py passes; "
            f"mypy reports no issue; coverage marker jobs-{1000 + i} is unchanged.\n"
        ),
        320,
        460,
    )

    long_trace = _extend_to(
        tokenizer,
        """Diagnose the transaction-resume failure described below. Produce a patch plan,
the exact invariants each change preserves, and the tests that would catch regression.
Do not attribute a root cause unless the evidence proves it. Do not change persistence
format, public APIs, dependency versions, or concurrency policy.

Service outline:
```python
def resume(transaction_id, repository, executor):
    saved = repository.load(transaction_id)
    branch = executor.restore(saved.snapshot)
    for event in saved.pending_events:
        branch.apply(event)
    return executor.run(branch)
```

All incidents were collected from the same build and are listed in chronological order.
""",
        lambda i: (
            f"incident {i:03d} | worker=exec-{i % 8} | tx=TX-{70000 + i} | "
            f"checkpoint={i * 16} | pending={(i % 5) + 1} | seed={i % 3}\n"
            f"  File \"src/runtime/resume.py\", line 84, in resume\n"
            f"  File \"src/runtime/executor.py\", line 219, in run\n"
            f"  observed=StateMismatch(expected_generation={i}, actual_generation={max(0, i - 1)})\n"
            f"  repository_digest=sha256:{hashlib.sha256(str(i).encode()).hexdigest()}\n"
        ),
        2300,
        3000,
    )
    long_diff = _extend_to(
        tokenizer,
        """Review this monorepo change request. Return a corrected unified diff only.
Requirements: preserve CLI output, keep serialization backward-compatible, reject unknown
configuration keys, and add focused tests. Files not shown are out of scope.

The proposed patch was assembled from multiple packages and may contain repeated mistakes.
""",
        lambda i: (
            f"diff --git a/packages/module_{i:03d}/config.py b/packages/module_{i:03d}/config.py\n"
            f"index {i:07x}..{i + 1:07x} 100644\n"
            f"--- a/packages/module_{i:03d}/config.py\n"
            f"+++ b/packages/module_{i:03d}/config.py\n"
            f"@@ -{10 + i},3 +{10 + i},4 @@ def load_config(raw):\n"
            f"-    timeout = raw.get(\"timeout\", 30)\n"
            f"+    timeout = raw.pop(\"timeout\", None)\n"
            f"+    timeout = timeout or 30\n"
            f"     return Config(timeout=timeout)\n"
            f"# CI note {i:03d}: unknown_key_{i % 9}=true must remain fatal; snapshot v{i % 4} must load.\n\n"
        ),
        2400,
        3200,
    )
    return (
        ("short_error_assertion", "short", short_error),
        ("short_diff_review", "short", short_diff),
        ("medium_cache_regression", "medium", medium_cache),
        ("medium_scoped_diff", "medium", medium_diff),
        ("long_resume_incidents", "long", long_trace),
        ("long_monorepo_diff", "long", long_diff),
    )


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    records = []
    for prompt_id, length_class, text in candidates(tokenizer):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lower, upper = {
            "short": (8, 32), "medium": (256, 512), "long": (2000, 4000)
        }[length_class]
        if not lower <= len(token_ids) <= upper:
            raise RuntimeError(f"{prompt_id}: {len(token_ids)} not in {lower}-{upper}")
        records.append(
            {
                "id": prompt_id,
                "length_class": length_class,
                "exact_token_count": len(token_ids),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PROPOSED_NOT_FROZEN",
        "tokenizer": {
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": CHECKPOINT_COMMIT,
            "add_special_tokens": False,
        },
        "prompts": records,
    }
    OUTPUT_PATH.write_text(
        yaml.dump(payload, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    for record in records:
        print(f"{record['id']}: {record['exact_token_count']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
