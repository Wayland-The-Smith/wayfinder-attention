#!/usr/bin/env python3
"""
Sanity-check pointer-scatter localization gap datasets.

With 0 decoy keys, pointer_unique has one needle segment, so scatter_multi_needles
does not activate multi-segment scatter (generator requires len(segments) > 1).
The task is classic single-needle NIAH at variable depth in synthetic noise — the
regime where dense reached 95% @ T=2048 (feasibility ladder).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.generator import (
    LongContextSampleGenerator,
    _needle_spans,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.holdout import resolve_holdout_splits
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    ACTIVE_TAG,
    trace_task_answer,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

CONFIG_DIR = ROOT / "configs" / "dense_linear_localization_gap"
PRIMARY = "pointer_scatter_t2048_0decoy"

EXPERIMENTS: dict[str, Path] = {
    "pointer_scatter_t1024_0decoy": CONFIG_DIR / "pointer_scatter_t1024_0decoy.yaml",
    "pointer_scatter_t2048_0decoy": CONFIG_DIR / "pointer_scatter_t2048_0decoy.yaml",
    "pointer_scatter_t4096_0decoy": CONFIG_DIR / "pointer_scatter_t4096_0decoy.yaml",
}


def _haystack_text(gen: LongContextSampleGenerator, sample) -> str:
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    suffix = sample.question
    if gen.config.include_answer_in_suffix:
        suffix = (
            f"{gen.config.question_prefix}{sample.question}"
            f"{gen.config.answer_prefix}{sample.expected_answer}"
        )
    ss = int(sample.meta_dict["suffix_start"])
    return text[:ss] + text[ss + len(suffix) :]


def _gold_key_rows(haystack: str, key: str) -> int:
    """Count authoritative pointer rows for key (not ACTIVE-tagged)."""
    import re

    pat = re.compile(rf"(?<!{re.escape(ACTIVE_TAG)} ){re.escape(key)} \S+\.")
    return len(pat.findall(haystack))


def verify_experiment(name: str, path: Path) -> None:
    arena = load_routing_arena_config(path)
    exp = build_arena_experiment_config(arena, dry_run=False)
    train_t = int(arena["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(exp, train_t)

    assert bench.task_types == ["pointer_unique"], bench.task_types
    assert bench.scatter_multi_needles is True
    assert bench.synthetic_decoy_keys == 0
    assert bench.num_distractors == 0
    assert bench.train_label_mode == "answer_only"
    assert bench.include_answer_in_suffix is True
    assert bench.seed != bench.holdout_seed

    payload = generate_synthetic_task(
        "pointer_unique",
        random.Random(0),
        {"synthetic_decoy_keys": 0, "answer_digit_width": bench.answer_digit_width},
    )
    assert len(payload.needle_segments) == 1, "0 decoys => exactly one needle row"
    assert payload.metadata.get("num_distractors") == 0
    probe = " ".join(payload.needle_segments)
    assert trace_task_answer(probe, payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    depths: list[float] = []
    scatter_flags: list[bool] = []

    for seed in range(12):
        sample = gen.generate_one(
            context_length=train_t,
            needle_depth=0.1 + 0.07 * (seed % 5),
            task_type="pointer_unique",
            haystack_mode="synthetic_noise",
            seed=seed,
        )
        assert sample.context_length == train_t
        assert sample.task_type == "pointer_unique"
        meta = sample.meta_dict
        assert meta.get("label_mode") == "answer_only"
        assert meta.get("include_answer_in_suffix") is True
        assert len(meta.get("needle_segments") or []) == 1
        assert meta.get("metadata", {}).get("num_distractors") == 0

        haystack = _haystack_text(gen, sample)
        traced = trace_task_answer(haystack, sample)
        assert traced == sample.expected_answer, f"trace mismatch seed={seed}"

        key = str(meta.get("metadata", {}).get("key", ""))
        assert key, "missing target key in metadata"
        assert _gold_key_rows(haystack, key) == 1, (
            f"expected exactly one gold row for key {key!r}, seed={seed}"
        )

        scatter_flags.append(bool(meta.get("scatter_needles")))
        depths.append(float(meta.get("needle_depth", -1)))

        spans = _needle_spans(haystack, meta.get("needle_segments") or [])
        assert len(spans) == 1, f"needle must appear once in haystack, seed={seed}"

    # 0 decoys: generator sets scatter=False (single segment); depth placement still varies.
    assert not any(scatter_flags), (
        "with 0 decoys, scatter_needles should be False (classic depth-NIAH, not multi-scatter)"
    )
    assert len(set(round(d, 2) for d in depths)) >= 3, "needle depths should vary across seeds"

    _, holdout_full, holdout_meta = resolve_holdout_splits(exp, bench, train_t)
    assert holdout_meta["holdout_full_samples"] == len(holdout_full)
    assert holdout_meta["train_disjoint_from_holdout"] is True
    for sample in holdout_full[:5]:
        assert sample.task_type == "pointer_unique"
        assert sample.context_length == train_t
        haystack = _haystack_text(gen, sample)
        assert trace_task_answer(haystack, sample) == sample.expected_answer

    print(f"  {name}: OK")
    print(f"    T={train_t}  task=pointer_unique  decoys=0  segments=1")
    print(f"    layout=classic depth-NIAH (scatter_needles=False with 1 segment)")
    print(f"    train_seed={bench.seed}  holdout_seed={bench.holdout_seed}")
    print(f"    holdout={len(holdout_full)} samples (target {holdout_meta['holdout_total_target']})  depth_range=[{min(depths):.2f}, {max(depths):.2f}]")
    if name == PRIMARY:
        s0 = gen.generate_one(
            context_length=train_t,
            needle_depth=0.5,
            task_type="pointer_unique",
            haystack_mode="synthetic_noise",
            seed=0,
        )
        key = s0.meta_dict["metadata"]["key"]
        val = s0.expected_answer
        print(f"    example: {key} {val}.  Q={s0.question!r}  answer={s0.expected_answer!r}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    print("=== verify_dense_linear_localization_gap ===")
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment {name!r}")
        verify_experiment(name, EXPERIMENTS[name])
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
