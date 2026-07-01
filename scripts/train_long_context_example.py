#!/usr/bin/env python3
"""Example training loop on procedural long-context retrieval data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, init_experiment_runtime, load_experiment_config
from routing_attention.benchmarks.long_context import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    get_long_context_dataloader,
    transfer_batch_to_device,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--attention", type=str, default="routing")
    args = parser.parse_args()

    config = load_experiment_config(7)
    device = init_experiment_runtime(config)
    config["model"]["attention_type"] = args.attention
    config["model"]["max_seq_len"] = 4096

    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    train_t = args.context_length
    config["model"]["max_seq_len"] = max(config["model"]["max_seq_len"], train_t)

    model = build_transformer(config, attention_type=args.attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = get_long_context_dataloader(
        bench_cfg,
        split="train",
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
        train_context_length=train_t,
    )
    data_iter = iter(loader)

    model.train()
    for step in range(args.steps):
        batch = transfer_batch_to_device(
            next(data_iter), device, pin_memory=device.type == "cuda"
        )
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch.get("attention_mask")

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attn_mask=attn_mask)
        logits = out["logits"][:, :-1].contiguous()
        target = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    print("Training example complete.")


if __name__ == "__main__":
    main()
