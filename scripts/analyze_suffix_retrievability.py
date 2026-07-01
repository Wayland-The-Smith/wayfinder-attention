#!/usr/bin/env python3
"""Analyze what fraction of NIAH samples are causally retrievable."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import _is_causally_reachable
from routing_attention.benchmarks.long_context.holdout import clear_holdout_cache, get_holdout_grid
from routing_attention.benchmarks.long_context.tokenizer import BenchmarkTokenizer


def _text(sample) -> str:
    tok = BenchmarkTokenizer(128)
    return tok.decode(sample.ids_np.tolist())


def _is_reachable(sample) -> bool:
    text = _text(sample)
    q = int(sample.meta_dict["question_start"])
    key = str(sample.metadata["key"])
    value = str(sample.metadata["value"])
    gold_pattern = sample.metadata.get("gold_needle")
    return _is_causally_reachable(
        text,
        q,
        key,
        value,
        gold_pattern=str(gold_pattern) if gold_pattern else None,
    )


def _suffix_depth(sample) -> float:
    return float(sample.meta_dict.get("suffix_depth", 0.0))


def holdout_stats(placement: str, decoys: int) -> None:
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        needle_depths=[0.1, 0.25, 0.5, 0.75, 0.9],
        task_types=["pointer_unique"],
        suffix_placement=placement,
        suffix_depth_min=0.1,
        suffix_depth_max=0.9,
        synthetic_decoy_keys=decoys,
        scatter_multi_needles=False,
        eval_samples_per_cell=20,
        holdout_seed=1000042,
    ).apply_synthetic_profile()
    clear_holdout_cache()
    samples = get_holdout_grid(cfg)
    clear_holdout_cache()

    impossible = sum(1 for s in samples if not _is_reachable(s))
    dists = [int(s.meta_dict["query_to_needle_distance"]) for s in samples]
    suffix_depths = [_suffix_depth(s) for s in samples]
    by_depth: dict[float, list[int]] = {}
    for s in samples:
        d = round(float(s.needle_depth), 2)
        by_depth.setdefault(d, [0, 0])
        if _is_reachable(s):
            by_depth[d][0] += 1
        else:
            by_depth[d][1] += 1

    print(f"HOLDOUT placement={placement!r} decoys={decoys} n={len(samples)}")
    print(
        f"  impossible (gold needle at/after query): {impossible}/{len(samples)} "
        f"= {100 * impossible / len(samples):.1f}%"
    )
    print(
        f"  query-needle distance: mean={sum(dists) / len(dists):.0f} "
        f"min={min(dists)} max={max(dists)}"
    )
    print(
        f"  suffix depth: mean={sum(suffix_depths) / len(suffix_depths):.2f} "
        f"min={min(suffix_depths):.2f} max={max(suffix_depths):.2f}"
    )
    for dep in sorted(by_depth):
        ok, bad = by_depth[dep]
        print(f"    needle_depth~{dep}: reachable={ok} impossible={bad}")


if __name__ == "__main__":
    holdout_stats("after_needles", 0)
    holdout_stats("at_end", 0)
    holdout_stats("random", 0)
    holdout_stats("random", 2)
