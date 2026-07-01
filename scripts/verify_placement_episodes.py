#!/usr/bin/env python3
"""Verify placement-episode training vs independent holdout."""

from __future__ import annotations

import sys

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import LongContextTrainDataset
from routing_attention.benchmarks.long_context.holdout import get_holdout_grid


def main() -> int:
    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[2048],
        task_types=["massive_addr_val"],
        num_kv_pairs=2,
        num_distractors=0,
        placement_episode_batches=3,
        scatter_placement_min=0,
        scatter_placement_max=200,
        seed=42,
        holdout_seed=1_000_042,
    ).apply_synthetic_profile()

    train_ds = LongContextTrainDataset(cfg, batch_size=4, train_context_length=2048)
    it = iter(train_ds)

    # One episode = 3 batches
    b0 = next(it)
    b1 = next(it)
    b2 = next(it)
    b3 = next(it)

    def episode_key(batch):
        metas = batch["meta"]
        return (
            metas[0]["question"],
            metas[0]["expected_answer"],
            tuple(metas[0].get("needle_segments") or []),
        )

    k0, k1, k2, k3 = episode_key(b0), episode_key(b1), episode_key(b2), episode_key(b3)
    assert k0 == k1 == k2, f"episode batches should share content: {k0!r} vs {k2!r}"
    assert k0 != k3, "fourth batch should start a new episode"
    assert b0["input_ids"].equal(b1["input_ids"]) is False, "layouts should differ within episode"
    print("ok: 3 batches share question/needles; batch 4 is new episode; layouts differ")

    holdout_cfg = cfg.holdout_config()
    assert holdout_cfg.placement_episode_batches == 0
    holdout = get_holdout_grid(holdout_cfg)
    holdout = [s for s in holdout if s.context_length == 2048]
    questions = [s.question for s in holdout]
    uniq_q = len(set(questions))
    print(
        f"ok: holdout {len(holdout)} samples, episode_batches=0, "
        f"unique_questions={uniq_q}/{len(holdout)} (procedural grid)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
