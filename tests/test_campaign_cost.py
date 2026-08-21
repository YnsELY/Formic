from __future__ import annotations

import runpy
from pathlib import Path


def _estimator():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "estimate_step2_campaign.py")
    )


def test_corrected_main_campaign_cost_is_stable():
    values = _estimator()
    prompts = values["PROMPTS"]
    aggregate = values["_aggregate"]
    cost_type = values["Cost"]

    total = cost_type(0, 0, 0, "")
    for length_class in ("short", "medium", "long"):
        selected = tuple(p for p in prompts if p.length_class == length_class)
        total += aggregate(selected, "prefill_full")
        segmentations = ("median", "quarters") if length_class == "long" else (
            "early", "median", "late", "quarters"
        )
        for segmentation in segmentations:
            total += aggregate(selected, "prefill_segmented", segmentation)
        decode_prompt = (selected[0],)
        total += aggregate(decode_prompt, "decode_cached")
        if length_class != "long":
            total += aggregate(decode_prompt, "decode_recompute")

    assert total.forwards == 1_416
    assert round(total.transfer_bytes / 2**30, 2) == 58.35


def test_final_session_includes_preflight_snapshot_and_accumulation_probe():
    values = _estimator()
    cost_type = values["Cost"]
    phases = values["_phase_costs"]((1_416, 0, 0))
    by_name = {name: cost for name, cost, _ in phases}
    total = sum((cost for _, cost, _ in phases), cost_type(0, 0, 0, ""))

    assert by_name["Préflight chronométré"].forwards == 207
    assert by_name["Snapshot/restore réel"].forwards == 48
    assert by_name["Sonde accumulation 64"].forwards == 1_280
    assert total.forwards == 4_139
