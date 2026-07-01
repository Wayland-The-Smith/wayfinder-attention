#!/usr/bin/env python3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    load_routing_arena_config,
    _resolve_synthetic_bench_cfg,
)

arena = load_routing_arena_config(ROOT / "configs/routing_arena_pointer_active_t2048.yaml")
cfg = build_arena_experiment_config(arena, dry_run=False)
bench = _resolve_synthetic_bench_cfg(cfg, 2048)
gen = LongContextSampleGenerator(bench)

t0 = time.perf_counter()
for i in range(10):
    s = gen.generate_one(
        context_length=2048,
        needle_depth=0.5,
        task_type="pointer_active",
        seed=1000 + i,
    )
    assert s.metadata.get("num_distractors") == 14, s.metadata
print(f"10 samples in {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
jobs = []
for depth in bench.needle_depths:
    for i in range(60):
        jobs.append((2048, depth, 1000042 + i + int(depth * 1000)))
for ctx, depth, seed in jobs:
    gen.generate_one(context_length=ctx, needle_depth=depth, task_type="pointer_active", seed=seed)
print(f"300 holdout-scale in {time.perf_counter() - t0:.1f}s")
