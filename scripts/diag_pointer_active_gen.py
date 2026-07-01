#!/usr/bin/env python3
"""Diagnose pointer_active at_end generation speed/failures."""
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator

cfg = LongContextBenchmarkConfig(
    context_lengths=[2048],
    task_types=["pointer_active"],
    needle_depths=[0.5],
    suffix_placement="at_end",
    num_distractors=14,
    benchmark_family="synthetic",
    seed=42,
    holdout_seed=1000042,
).apply_synthetic_profile()
gen = LongContextSampleGenerator(cfg)

t0 = time.perf_counter()
s = gen.generate_one(
    context_length=2048,
    needle_depth=0.5,
    task_type="pointer_active",
    seed=1,
)
print(f"one sample ok distractors={s.metadata['num_distractors']} sec={time.perf_counter()-t0:.3f}")

errors: Counter = Counter()
ok = 0
t0 = time.perf_counter()
for i in range(20):
    t1 = time.perf_counter()
    try:
        gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type="pointer_active",
            seed=100 + i,
        )
        ok += 1
        print(f"  sample {i} ok {time.perf_counter()-t1:.3f}s", flush=True)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        errors[msg[:100]] += 1
        print(f"  sample {i} FAIL {msg[:80]} {time.perf_counter()-t1:.3f}s", flush=True)
print(f"batch ok={ok}/20 total={time.perf_counter()-t0:.1f}s errors={dict(errors)}")
