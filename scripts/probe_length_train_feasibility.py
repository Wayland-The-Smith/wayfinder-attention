#!/usr/bin/env python3
"""
Probe train-step latency + VRAM at long T (largest first) for dense_flash.

Used to pick n_layers before length-scaling training sweep.

Usage:
  python scripts/probe_length_train_feasibility.py
  python scripts/probe_length_train_feasibility.py --lengths 16384,8192 --layers 4,2,1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, set_seed
from experiments.experiment_4 import _expand_position_embeddings
from experiments.experiment_7 import (
    _effective_train_batch_size,
    _weighted_long_context_loss,
)
from paper_evidence_common import (
    PAPER_OUTPUT,
    MAX_TRAIN_HOURS_10K,
    MAX_TRAIN_STEP_MS,
    build_cell_config,
    projected_hours,
    train_feasible,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    get_long_context_dataloader,
    transfer_batch_to_device,
)
from routing_attention.benchmarks.long_context.routing_arena import init_arena_runtime
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.cuda import configure_cuda_training


VARIANT_ATTN = {
    "dense_flash": "dense_flash",
    "linear": "linear",
    "local_window64": "local",
    "local_window256": "local",
    "key_vector_k32": "key_vector",
    "learned_address_k32": "learned_address",
}


def probe_train_step(
    *,
    train_t: int,
    n_layers: int,
    variant: str = "dense_flash",
    warmup: int = 2,
    timed: int = 5,
) -> dict:
    device = torch.device("cuda")
    config = build_cell_config(
        train_t=train_t,
        decoys=0,
        dry_run=False,
        n_layers=n_layers,
        seed=45,
        tag=f"probe_T{train_t}",
        train_steps=100,
    )
    init_arena_runtime(config)
    set_seed(45, deterministic=True)

    attn = VARIANT_ATTN.get(variant, variant)
    if variant == "local_window64":
        config.setdefault("router", {})["local_window"] = 64
        config.setdefault("routing_attention", {})["local_window"] = 64
    elif variant == "local_window256":
        config.setdefault("router", {})["local_window"] = 256
        config.setdefault("routing_attention", {})["local_window"] = 256

    model = build_transformer(config, attention_type=attn).to(device)
    _expand_position_embeddings(model, train_t)
    model.train()

    batch_size = _effective_train_batch_size(
        int(config.get("data", {}).get("batch_size", 4)),
        train_t,
        attn,
    )
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
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
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    reset_peak_vram(device)
    times: list[float] = []
    oom = False
    err = ""

    try:
        for step in range(warmup + timed):
            batch = transfer_batch_to_device(next(data_iter), device, pin_memory=True)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
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
            if step >= warmup:
                times.append(time.perf_counter() - t0)
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        err = str(exc)[:300]

    step_ms = 1000.0 * sum(times) / len(times) if times else None
    peak_mb = peak_vram_mb(device)

    return {
        "context_length": train_t,
        "n_layers": n_layers,
        "variant": variant,
        "batch_size": batch_size,
        "step_ms_mean": round(step_ms, 2) if step_ms else None,
        "peak_vram_mb": round(peak_mb, 1) if peak_mb else None,
        "oom": oom,
        "error": err,
        "est_10k_hours": round(projected_hours(step_ms, 10_000) or 0, 2) if step_ms else None,
        "feasible": train_feasible(step_ms, 10_000) if step_ms else False,
    }


def recommend_layers(probe_rows: list[dict], train_t: int) -> dict:
    """Pick largest layer count feasible at train_t for dense."""
    candidates = [4, 2, 1]
    for n in candidates:
        rows = [r for r in probe_rows if r["context_length"] == train_t and r["n_layers"] == n and r["variant"] == "dense_flash"]
        if rows and rows[0].get("feasible"):
            return {"train_t": train_t, "n_layers": n, "probe": rows[0]}
    # fallback: lowest layer count that didn't OOM
    for n in reversed(candidates):
        rows = [r for r in probe_rows if r["context_length"] == train_t and r["n_layers"] == n and r["variant"] == "dense_flash"]
        if rows and not rows[0].get("oom"):
            return {"train_t": train_t, "n_layers": n, "probe": rows[0], "fallback": True}
    return {"train_t": train_t, "n_layers": None, "feasible": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="16384,8192,4096,2048")
    parser.add_argument("--layers", default="4,2,1")
    parser.add_argument("--variants", default="dense_flash")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    configure_cuda_training({"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}})
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    device = torch.device("cuda")
    info = collect_device_info(device)
    info.update(backend_status())

    print("=== length train-step feasibility probe ===")
    print(f"  max_step_ms={MAX_TRAIN_STEP_MS}  max_10k_hours={MAX_TRAIN_HOURS_10K}")
    print(json.dumps(info, indent=2))
    print()

    results: list[dict] = []
    # Longest T first; stop layer sweep at a T if all layers OOM
    for train_t in lengths:
        print(f"--- T={train_t} ---")
        all_oom = True
        for n_layers in layers:
            for variant in variants:
                print(f"  probe {variant} {n_layers}L T={train_t} ...", flush=True)
                row = probe_train_step(train_t=train_t, n_layers=n_layers, variant=variant)
                results.append(row)
                status = "OOM" if row["oom"] else f"{row['step_ms_mean']} ms/step"
                print(
                    f"    {variant} {n_layers}L: {status}  vram={row['peak_vram_mb']}MB  "
                    f"10k≈{row['est_10k_hours']}h  feasible={row['feasible']}"
                )
                if not row["oom"]:
                    all_oom = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if all_oom and variants == ["dense_flash"]:
            print(f"  all dense probes OOM at T={train_t}; skipping longer contexts")
            break

    recommendations = [recommend_layers(results, t) for t in lengths]
    out = {
        "device": info,
        "thresholds": {"max_step_ms": MAX_TRAIN_STEP_MS, "max_10k_hours": MAX_TRAIN_HOURS_10K},
        "probes": results,
        "recommendations": recommendations,
    }

    out_path = Path(args.output) if args.output else PAPER_OUTPUT / "feasibility_probe" / "train_step_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("Recommendations:")
    for rec in recommendations:
        n = rec.get("n_layers")
        fb = " (fallback)" if rec.get("fallback") else ""
        print(f"  T={rec['train_t']}: n_layers={n}{fb}")


if __name__ == "__main__":
    main()
