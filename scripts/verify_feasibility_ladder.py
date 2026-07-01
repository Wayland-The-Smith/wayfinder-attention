#!/usr/bin/env python3
"""Structural checks for NIAH feasibility ladder (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.feasibility_ladder import (
    _build_overfit_holdout,
    _merge_level_config,
    _resolve_bench_cfg,
    evaluate_pass_criteria,
    load_feasibility_ladder_config,
)


def test_config_loads():
    cfg = load_feasibility_ladder_config()
    levels = cfg.get("levels", [])
    assert len(levels) == 5, f"expected 5 levels, got {len(levels)}"
    assert [int(l["level"]) for l in levels] == [0, 1, 2, 3, 4]
    print("config loads OK (5 levels)")


def test_level0_suffix_after_needles():
    ladder = load_feasibility_ladder_config()
    level = ladder["levels"][0]
    config = _merge_level_config(ladder, level, dry_run=True)
    bench = _resolve_bench_cfg(config, level)
    assert bench.suffix_placement == "after_needles"
    assert bench.overfit_train_samples == 32
    assert bench.task_types == ["pointer_unique"]
    assert bench.synthetic_decoy_keys == 0
    print("level 0 benchmark overrides OK")


def test_level0_overfit_holdout():
    ladder = load_feasibility_ladder_config()
    level = ladder["levels"][0]
    config = _merge_level_config(ladder, level, dry_run=False)
    bench = _resolve_bench_cfg(config, level)
    holdout = _build_overfit_holdout(bench, 512)
    assert len(holdout) == 32
    depths = {float(s.needle_depth) for s in holdout}
    assert depths, "needle depths missing"
    for s in holdout:
        assert s.task_type == "pointer_unique"
        assert len(s.expected_answer) == 1, "L0 single-token answer expected"
    print("level 0 overfit holdout OK (32 samples, single-char answers)")


def test_level1_distance_isolated():
    ladder = load_feasibility_ladder_config()
    level = next(l for l in ladder["levels"] if int(l["level"]) == 1)
    assert level.get("save_checkpoint") is False
    config = _merge_level_config(ladder, level, dry_run=False)
    bench = _resolve_bench_cfg(config, level)
    assert bench.task_types == ["pointer_unique"]
    assert bench.suffix_placement == "after_needles"
    assert bench.scatter_multi_needles is False
    assert bench.synthetic_decoy_keys == 0
    assert 2048 in bench.context_lengths
    print("level 1 (local sanity after_needles) OK")


def test_level2_classic_niah():
    ladder = load_feasibility_ladder_config()
    level = next(l for l in ladder["levels"] if int(l["level"]) == 2)
    assert level.get("save_checkpoint") is True
    config = _merge_level_config(ladder, level, dry_run=False)
    bench = _resolve_bench_cfg(config, level)
    assert bench.task_types == ["pointer_unique"]
    assert bench.suffix_placement == "at_end"
    assert bench.scatter_multi_needles is False
    assert bench.synthetic_decoy_keys == 0
    cal = config.get("dense_calibration", {})
    assert cal.get("restore_best_checkpoint") is True
    assert cal.get("eval_use_full_holdout") is True
    print("level 2 (classic NIAH dense baseline gate) OK")


def test_level3_full_gate():
    ladder = load_feasibility_ladder_config()
    level = next(l for l in ladder["levels"] if int(l["level"]) == 3)
    assert int(level["requires_level"]) == 2
    config = _merge_level_config(ladder, level, dry_run=False)
    bench = _resolve_bench_cfg(config, level)
    assert bench.suffix_placement == "at_end"
    assert len(bench.context_curriculum) == 3
    assert bench.context_curriculum[0]["context_length"] == 512
    assert bench.context_curriculum[-1]["context_length"] == 8192
    print("level 3 (full gate at_end + curriculum) OK")


def test_level4_requires_checkpoint_level():
    ladder = load_feasibility_ladder_config()
    level = next(l for l in ladder["levels"] if int(l["level"]) == 4)
    assert int(level["requires_level"]) == 3
    assert level.get("variants")
    config = _merge_level_config(ladder, level, dry_run=False)
    bench = _resolve_bench_cfg(config, level)
    assert bench.suffix_placement == "at_end"
    print("level 4 (variant breakthrough) OK")


def test_pass_criteria_eval():
    level = {
        "pass_criteria": {
            "primary_gate_accuracy_min": 0.90,
            "by_task_type": {"pointer_unique": 0.80},
        }
    }
    ok = evaluate_pass_criteria(
        level,
        {"primary_gate_accuracy": 0.95, "by_task_type": {"pointer_unique": 0.85}},
    )
    assert ok["passed"]
    bad = evaluate_pass_criteria(
        level,
        {"primary_gate_accuracy": 0.50, "by_task_type": {"pointer_unique": 0.85}},
    )
    assert not bad["passed"]
    print("pass criteria evaluation OK")


def test_dry_run_step_caps():
    ladder = load_feasibility_ladder_config()
    for level in ladder["levels"]:
        config = _merge_level_config(ladder, level, dry_run=True)
        steps = int(config["transformer"]["max_steps"])
        is_variant = bool(level.get("variants"))
        cap = 30 if is_variant else 50
        assert steps <= cap, f"level {level['level']} dry_run steps={steps}"
    print("dry_run step caps OK")


def main():
    test_config_loads()
    test_level0_suffix_after_needles()
    test_level0_overfit_holdout()
    test_level1_distance_isolated()
    test_level2_classic_niah()
    test_level3_full_gate()
    test_level4_requires_checkpoint_level()
    test_pass_criteria_eval()
    test_dry_run_step_caps()
    print("\nAll feasibility ladder structural checks passed.")


if __name__ == "__main__":
    main()
