#!/usr/bin/env python3
"""
Profile Experiment 7 training: data generation vs GPU compute per variant.

Measures the exact training-step path used in experiments/experiment_7.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import (
    build_transformer,
    init_experiment_runtime,
    load_experiment_config,
    load_router_from_reuse,
    load_addresses_from_reuse,
)
from experiments.experiment_4 import _apply_variant_config, _expand_position_embeddings
from experiments.experiment_7 import (
    EXP7_ATTENTION_VARIANTS,
    _effective_train_batch_size,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    LongContextTrainDataset,
    get_long_context_dataloader,
    transfer_batch_to_device,
)
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_setup import apply_routing_variant_settings
from routing_attention.models.fast_attention import backend_status


VARIANTS = [
    "dense_flash",
    "linear",
    "local_window64",
    "local_window256",
    "routing_asymmetric",
    "learned_address_k32",
    "key_vector_k32",
]


def _build_variant(variant: str, config: dict, device: torch.device, max_seq: int):
    var_config = _apply_variant_config(config.copy(), variant)
    var_config["model"]["max_seq_len"] = max_seq
    attn_type = var_config.get("model", {}).get("attention_type") or EXP7_ATTENTION_VARIANTS.get(
        variant, "routing"
    )
    var_config.setdefault("model", {})["attention_type"] = attn_type

    if attn_type == "routing":
        router, _ = load_router_from_reuse(var_config, device)
        model = build_transformer(var_config, attention_type="routing", router=router).to(device)
    else:
        model = build_transformer(var_config, attention_type=attn_type).to(device)

    if attn_type == "learned_address":
        load_addresses_from_reuse(var_config, device, model=model)

    _expand_position_embeddings(model, max_seq)
    apply_routing_variant_settings(model, var_config, attn_type, max_seq)
    return model, var_config, attn_type


def _train_step(model, batch, device, use_amp: bool) -> None:
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    batch = transfer_batch_to_device(batch, device, non_blocking=True, pin_memory=True)
    with amp_ctx:
        out = model(input_ids=batch["input_ids"], attn_mask=batch.get("attention_mask"))
        logits = out["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    loss.backward()


def bench_sample_gen(T: int, n: int = 20) -> float:
    gen = LongContextSampleGenerator(LongContextBenchmarkConfig())
    t0 = time.perf_counter()
    for i in range(n):
        gen.generate_one(
            context_length=T,
            needle_depth=0.5,
            task_type="exact_retrieval",
            haystack_mode="random_sentences",
            seed=i,
        )
    return (time.perf_counter() - t0) / n * 1000.0


def bench_dataloader(T: int, batch_size: int, num_workers: int, n: int = 10) -> float:
    bench_cfg = LongContextBenchmarkConfig()
    loader = get_long_context_dataloader(
        bench_cfg,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        train_context_length=T,
    )
    it = iter(loader)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        next(it)
        times.append(time.perf_counter() - t0)
    return 1000.0 * sum(times) / len(times)


def bench_variant(
    variant: str,
    T: int,
    *,
    warmup: int = 3,
    runs: int = 8,
    num_workers: int = 2,
) -> dict:
    config = load_experiment_config(7)
    device = init_experiment_runtime(config)
    use_amp = bool(config.get("training", {}).get("use_amp", True))

    model, var_config, attn_type = _build_variant(variant, config, device, T)
    bs = _effective_train_batch_size(
        int(config.get("data", {}).get("batch_size", 1)), T, attn_type
    )

    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    loader = get_long_context_dataloader(
        bench_cfg,
        split="train",
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=True,
        train_context_length=T,
    )
    data_iter = iter(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup (includes first-batch generation + kernel compile)
    for _ in range(warmup):
        batch = next(data_iter)
        _train_step(model, batch, device, use_amp)
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

    data_ms: list[float] = []
    compute_ms: list[float] = []
    total_ms: list[float] = []

    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        batch = next(data_iter)
        if device.type == "cuda":
            torch.cuda.synchronize()
        data_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _train_step(model, batch, device, use_amp)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_ms.append((time.perf_counter() - t1) * 1000.0)
        total_ms.append((time.perf_counter() - t_total) * 1000.0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "attn_type": attn_type,
        "T": T,
        "batch_size": bs,
        "data_ms": sum(data_ms) / len(data_ms),
        "compute_ms": sum(compute_ms) / len(compute_ms),
        "total_ms": sum(total_ms) / len(total_ms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    print("=== Attention backends ===")
    print(backend_status())
    print()

    print("=== CPU sample generation (single sample, ms) ===")
    for T in args.context_lens:
        ms = bench_sample_gen(T)
        print(f"  T={T:5d}  generate_one: {ms:8.1f} ms")
    print()

    print("=== Dataloader next-batch (batch=1, ms) ===")
    for T in args.context_lens:
        for nw in (0, 2):
            ms = bench_dataloader(T, batch_size=1, num_workers=nw)
            print(f"  T={T:5d}  num_workers={nw}: {ms:8.1f} ms/batch")
    print()

    print("=== Full training step (experiment_7 path, bf16 AMP) ===")
    print(f"{'variant':<22} {'T':>6} {'bs':>3} {'data_ms':>9} {'gpu_ms':>9} {'total_ms':>9} {'tok/s':>10}")
    print("-" * 78)
    rows = []
    for T in args.context_lens:
        for var in args.variants:
            try:
                r = bench_variant(var, T, runs=args.runs, num_workers=2)
                rows.append(r)
                tok_s = (T * r["batch_size"]) / (r["total_ms"] / 1000.0)
                print(
                    f"{r['variant']:<22} {T:6d} {r['batch_size']:3d} "
                    f"{r['data_ms']:9.1f} {r['compute_ms']:9.1f} {r['total_ms']:9.1f} {tok_s:10.0f}"
                )
            except torch.cuda.OutOfMemoryError as exc:
                print(f"{var:<22} {T:6d}   OOM: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"{var:<22} {T:6d}   ERROR: {exc}")

    print()
    if rows:
        dense = next((r for r in rows if r["variant"] == "dense_flash" and r["T"] == 32768), None)
        route = next((r for r in rows if r["variant"] == "routing_asymmetric" and r["T"] == 32768), None)
        if dense and route:
            print(
                f"T=32768 routing vs dense_flash GPU: "
                f"{route['compute_ms']:.0f} ms vs {dense['compute_ms']:.0f} ms "
                f"({dense['compute_ms']/route['compute_ms']:.2f}x)"
            )


if __name__ == "__main__":
    main()
