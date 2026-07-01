#!/usr/bin/env python3
"""Benchmark N-layer dense_flash training step latency vs context length (RTX 5090 probe)."""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, init_experiment_runtime, load_experiment_config, set_seed
from experiments.experiment_4 import _expand_position_embeddings
from experiments.experiment_7 import (
    _effective_train_batch_size,
    _weighted_long_context_loss,
    _benchmark_config_from_run,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    get_long_context_dataloader,
    transfer_batch_to_device,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile
from routing_attention.models.fast_attention import backend_status, warmup_fla_linear_kernels
from routing_attention.utils.config import load_config


def _build_config(train_t: int, *, n_layers: int = 2) -> dict:
    raw = load_experiment_config(7, variant="dense_flash")
    config = apply_suite_profile(raw, "fast")
    config["model"]["n_layers"] = int(n_layers)
    config["model"]["max_seq_len"] = max(train_t, int(config["model"].get("max_seq_len", 8192)))
    config["long_context_benchmark"] = {
        **config.get("long_context_benchmark", {}),
        **LongContextBenchmarkConfig(
            context_lengths=[train_t],
            task_types=["pointer_unique"],
            needle_depths=[0.5],
            suffix_placement="at_end",
            scatter_multi_needles=False,
            synthetic_decoy_keys=0,
            benchmark_family="synthetic",
        )
        .apply_synthetic_profile()
        .to_dict(),
    }
    return config


def benchmark_context_length(
    train_t: int,
    *,
    n_layers: int = 2,
    warmup_steps: int = 3,
    timed_steps: int = 20,
    batch_size_cfg: int = 4,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for benchmark")

    config = _build_config(train_t, n_layers=n_layers)
    init_experiment_runtime(config)
    set_seed(42)

    model = build_transformer(config, attention_type="dense_flash").to(device)
    _expand_position_embeddings(model, train_t)
    model.train()

    attn_type = config["model"].get("attention_type", "dense_flash")
    batch_size = _effective_train_batch_size(batch_size_cfg, train_t, attn_type)
    use_amp = True
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    bench_cfg = _benchmark_config_from_run(config)
    if bench_cfg.benchmark_family == "synthetic":
        bench_cfg = bench_cfg.apply_synthetic_profile()

    loader = get_long_context_dataloader(
        bench_cfg,
        split="train",
        batch_size=batch_size,
        num_workers=0,
        train_context_length=train_t,
    )
    data_iter = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    reset_peak_vram(device)
    oom = False
    error = ""

    try:
        for step in range(warmup_steps + timed_steps):
            batch = transfer_batch_to_device(next(data_iter), device, pin_memory=True)
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            with amp_ctx:
                out = model(input_ids=batch["input_ids"], attn_mask=batch.get("attention_mask"))
                logits = out["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()
                shift_weights = batch.get("loss_weights")
                if shift_weights is not None:
                    shift_weights = shift_weights[:, 1:].contiguous()
                loss = _weighted_long_context_loss(shift_logits, shift_labels, shift_weights)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if step >= warmup_steps:
                if step == warmup_steps:
                    times = [elapsed]
                else:
                    times.append(elapsed)
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        error = str(exc)
        times = []

    peak_mb = peak_vram_mb(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if times:
        mean_ms = 1000.0 * sum(times) / len(times)
        med_ms = 1000.0 * sorted(times)[len(times) // 2]
        steps_per_sec = 1000.0 / mean_ms
        est_8k_min = (8000 * mean_ms) / 1000.0 / 60.0
        est_50k_min = (50000 * mean_ms) / 1000.0 / 60.0
    else:
        mean_ms = med_ms = steps_per_sec = est_8k_min = est_50k_min = None

    return {
        "context_length": train_t,
        "n_layers": int(config["model"]["n_layers"]),
        "d_model": int(config["model"]["d_model"]),
        "batch_size": batch_size,
        "trainable_params": trainable,
        "oom": oom,
        "error": error[:200] if error else "",
        "step_ms_mean": round(mean_ms, 2) if mean_ms else None,
        "step_ms_median": round(med_ms, 2) if med_ms else None,
        "steps_per_sec": round(steps_per_sec, 3) if steps_per_sec else None,
        "peak_vram_mb": round(peak_mb, 1),
        "est_8k_steps_min": round(est_8k_min, 2) if est_8k_min else None,
        "est_50k_steps_min": round(est_50k_min, 1) if est_50k_min else None,
        "timed_steps": len(times),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    n_layers = int(args.n_layers)

    lengths = [2048, 4096, 8192, 16384, 32768]
    device = torch.device("cuda")
    print(f"=== {n_layers}L dense_flash train-step benchmark ===")
    info = collect_device_info(device)
    info.update(backend_status())
    print(json.dumps(info, indent=2))
    print()

    results: list[dict] = []
    for t in lengths:
        print(f"Benchmarking T={t} ...", flush=True)
        row = benchmark_context_length(t, n_layers=n_layers)
        results.append(row)
        status = "OOM" if row["oom"] else f"{row['step_ms_mean']} ms/step"
        print(
            f"  T={t:>5}  bs={row['batch_size']}  {status}  "
            f"vram={row['peak_vram_mb']}MB  "
            f"8k≈{row['est_8k_steps_min']}min  50k≈{row['est_50k_steps_min']}min"
        )
        if row["oom"]:
            print("  (stopping sweep — longer contexts likely worse)")
            break
        torch.cuda.empty_cache()

    default_dir = ROOT / "experiments" / "Experiment_7" / f"feasibility_ladder_{n_layers}L"
    out_path = Path(args.output) if args.output else default_dir / f"dense_{n_layers}l_step_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"device": info, "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
