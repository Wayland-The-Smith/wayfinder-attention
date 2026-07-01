#!/usr/bin/env python3
"""Quick diagnostics for massive_addr_val failures."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.holdout import get_holdout_grid
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_task_answer


def analyze_predictions(run_dir: Path) -> None:
    latest = json.loads((run_dir / "latest.json").read_text())
    ev = latest["results"]["dense_flash"].get("eval_official") or latest["results"]["dense_flash"].get("eval", {})
    recs = ev.get("records", [])
    if not recs:
        print(f"  {run_dir.name}: no per-sample records in JSON")
        return
    wrong = [r for r in recs if not r["correct"]]
    correct = [r for r in recs if r["correct"]]
    pred_counter = Counter(r["predicted"] for r in recs)
    print(f"\n=== {run_dir.name} ===")
    print(f"  correct={len(correct)}/{len(recs)}")
    print(f"  top predictions: {pred_counter.most_common(8)}")
    print(f"  sample wrong: ", [(r["expected"], r["predicted"]) for r in wrong[:6]])


def check_holdout_trace(num_kv_pairs: int) -> None:
    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[2048],
        task_types=["massive_addr_val"],
        num_kv_pairs=num_kv_pairs,
        num_distractors=0,
    ).apply_synthetic_profile()
    holdout = get_holdout_grid(cfg)
    fails = 0
    for s in holdout[:50]:
        text = cfg  # placeholder
        gen = LongContextSampleGenerator(cfg)
        text = gen.tokenizer.decode(s.ids_np.tolist())
        qpos = text.find(s.question)
        hay = text[:qpos] if qpos >= 0 else text
        from routing_attention.benchmarks.long_context.tasks import TaskPayload

        payload = TaskPayload(
            needle_segments=s.metadata.get("needle_segments", []),
            question=s.question,
            expected_answer=s.expected_answer,
            task_type=s.task_type,
            metadata=s.metadata,
        )
        traced = trace_task_answer(hay, payload)
        if traced != s.expected_answer:
            fails += 1
    print(f"\nholdout trace check N={num_kv_pairs}: {fails}/50 mismatches on first 50 samples")


def overfit_smoke(num_kv_pairs: int, steps: int = 500) -> None:
    """Train dense from scratch on 16 fixed samples — can it memorize?"""
    from experiments.experiment_7 import _train_on_benchmark
    from routing_attention.models.transformer import RoutingTransformer

    cfg_dict = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[2048],
        task_types=["massive_addr_val"],
        num_kv_pairs=num_kv_pairs,
        num_distractors=0,
        overfit_train_samples=16,
        seed=99,
    ).apply_synthetic_profile().to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 4,
        "vocab_size": 128,
        "max_seq_len": 2048,
        "attn_type": "dense_flash",
    }
    model = RoutingTransformer(**model_cfg).to(device)
    bench_cfg = LongContextBenchmarkConfig.from_dict(cfg_dict)

  # quick manual loop
    from routing_attention.benchmarks.long_context.dataset import get_long_context_dataloader
    from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator

    loader = get_long_context_dataloader(
        bench_cfg, split="train", batch_size=4, num_workers=0, train_context_length=2048
    )
    evaluator = LongContextEvaluator(bench_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    data_iter = iter(loader)
    for step in range(steps):
        batch = next(data_iter)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        opt.zero_grad()
        out = model(input_ids=batch["input_ids"], attn_mask=batch.get("attention_mask"))
        logits = out["logits"]
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        opt.step()
        if step in (0, steps - 1):
            correct = 0
            total = 0
            with torch.no_grad():
                for bi in range(batch["input_ids"].size(0)):
                    meta = batch["meta"][bi] if batch.get("meta") else None
                    if not meta:
                        continue
                    rec = evaluator.score_sample(logits[bi : bi + 1], meta)
                    total += 1
                    if rec.correct:
                        correct += 1
            print(
                f"  overfit N={num_kv_pairs} step={step+1} loss={loss.item():.3f} "
                f"batch_acc={correct}/{total}"
            )


def main() -> None:
    sweep = ROOT / "experiments/Experiment_7/massive_addr_val_capacity_sweep"
    for d in sorted(sweep.glob("N*_massive_addr_val_dense_flash")):
        analyze_predictions(d)

    check_holdout_trace(2)

    print("\n=== Overfit from scratch (16 fixed samples) ===")
    for n in [2, 10]:
        overfit_smoke(n, steps=400)


if __name__ == "__main__":
    main()
