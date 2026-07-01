#!/usr/bin/env python3
"""
Quick Experiment 7 suite profiler — exercises full three-stage protocol without full training.

Per (profile, T, variant):
  Stage A   — minimal dense_flash pretrain (few steps) → C_dense(T)
  Stage A.5 — router index from NIAH dense teacher (routing_asymmetric only)
  Stage B   — minimal sparse fine-tune with staged-training audit
  Forward latency + VRAM + wall-time estimates for full suite
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import init_experiment_runtime, load_experiment_config
from experiments.experiment_4 import _apply_variant_config
from experiments.experiment_7 import (
    EXP7_ATTENTION_VARIANTS,
    _build_variant_model,
    _effective_train_batch_size,
    _resolve_train_steps,
    _save_dense_checkpoint,
    _train_on_benchmark,
    _verify_staged_training_protocol,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.index_pretrain import (
    pretrain_router_on_dense_checkpoint,
    router_index_checkpoint_path,
)
from routing_attention.benchmarks.long_context.production_backends import assert_production_backends_available
from routing_attention.benchmarks.long_context.runtime import peak_vram_mb, reset_peak_vram
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile, dense_checkpoint_path
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.config import load_config, merge_configs

VARIANTS = [
    "dense_flash",
    "linear",
    "local_window64",
    "local_window256",
    "routing_asymmetric",
    "learned_address_k32",
    "key_vector_k32",
]


def _profile_config(profile: str, smoke_override: dict | None = None) -> dict:
    raw = load_config(ROOT / "configs" / "experiment_7.yaml")
    patched = apply_suite_profile(raw, profile)
    ovr = {
        "model": patched.get("model", {}),
        "transformer": patched.get("transformer", {}),
        "data": patched.get("data", {}),
        "evaluation": patched.get("evaluation", {}),
        "long_context_benchmark": patched.get("long_context_benchmark", {}),
        "index_pretrain": patched.get("index_pretrain", {}),
        "router": patched.get("router", {}),
        "data_collection": patched.get("data_collection", {}),
        "suite_active_profile": patched.get("suite_active_profile", {}),
    }
    cfg = merge_configs(load_experiment_config(7), ovr)
    if smoke_override:
        cfg = merge_configs(cfg, smoke_override)
    return cfg


def _minimal_train(
    model,
    var_config: dict,
    config: dict,
    device: torch.device,
    train_t: int,
    stage: str,
    n_steps: int,
) -> dict:
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    cfg = merge_configs(var_config, {"transformer": {"max_steps": n_steps}})
    t0 = time.perf_counter()
    info = _train_on_benchmark(
        model,
        cfg,
        bench_cfg,
        holdout_samples=[],
        device=device,
        train_context_length=train_t,
        logger=__import__("logging").getLogger("profile"),
        max_steps=n_steps,
        training_stage=stage,
    )
    info["wall_sec"] = time.perf_counter() - t0
    if n_steps > 0 and info.get("final_loss") is not None:
        info["mean_step_ms"] = info["wall_sec"] * 1000.0 / n_steps
    return info


def _latency_ms(model, config: dict, device: torch.device, train_t: int) -> float | None:
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=[])
    model.eval()
    result = evaluator.benchmark_forward_latency(
        model, device=device, context_length=train_t, warmup=1, runs=3
    )
    return result.get("latency_ms")


def profile_cell(
    config: dict,
    profile_name: str,
    variant: str,
    train_t: int,
    device: torch.device,
    ckpt_dir: Path,
    index_dir: Path,
    *,
    dense_smoke_steps: int,
    sparse_smoke_steps: int,
    index_smoke: bool,
) -> dict:
    row: dict = {
        "profile": profile_name,
        "variant": variant,
        "train_context_length": train_t,
        "status": "ok",
    }
    reset_peak_vram(device)
    t_total = time.perf_counter()

    try:
        dense_ckpt = dense_checkpoint_path(ckpt_dir, train_t)
        router_idx = router_index_checkpoint_path(index_dir, train_t)

        # Stage A — ensure minimal dense checkpoint
        if not dense_ckpt.exists():
            model, _ = _build_variant_model(config, "dense_flash", device, train_t)
            audit = _verify_staged_training_protocol(
                "dense_flash",
                "dense_pretrain",
                getattr(model, "_exp7_routing_info", {}),
                model,
                two_stage=True,
            )
            row["stage_a_audit"] = audit
            row["stage_a"] = _minimal_train(
                model, config, config, device, train_t, "dense_pretrain", dense_smoke_steps
            )
            _save_dense_checkpoint(
                model, dense_ckpt, train_context_length=train_t, trained_steps=dense_smoke_steps
            )
            del model
            reset_peak_vram(device)

        row["dense_checkpoint"] = str(dense_ckpt)

        # Stage A.5 — NIAH router index (routing_asymmetric)
        if variant == "routing_asymmetric" and index_smoke:
            if not router_idx.exists():
                meta = pretrain_router_on_dense_checkpoint(
                    config,
                    dense_ckpt,
                    train_t,
                    router_idx,
                    device,
                    dry_run=True,
                )
                row["stage_a5"] = meta
            row["router_index_checkpoint"] = str(router_idx)

        if variant == "dense_flash":
            stage = "dense_pretrain"
            init_ckpt = None
            router_ckpt = None
            smoke_steps = dense_smoke_steps
        else:
            stage = "finetune_from_dense"
            init_ckpt = dense_ckpt
            router_ckpt = router_idx if variant == "routing_asymmetric" else None
            smoke_steps = sparse_smoke_steps

        model, var_config = _build_variant_model(
            config,
            variant,
            device,
            train_t,
            dense_checkpoint=init_ckpt,
            router_index_checkpoint=router_ckpt,
        )
        routing_info = getattr(model, "_exp7_routing_info", {})
        audit = _verify_staged_training_protocol(
            variant, stage, routing_info, model, two_stage=True
        )
        row["training_audit"] = audit
        row["routing_setup"] = routing_info

        row["stage_b"] = _minimal_train(
            model, var_config, config, device, train_t, stage, smoke_steps
        )
        row["peak_vram_mb"] = peak_vram_mb(device)
        row["latency_ms"] = _latency_ms(model, config, device, train_t)

        attn_type = routing_info.get("attn_type") or EXP7_ATTENTION_VARIANTS.get(variant, "routing")
        bs = _effective_train_batch_size(int(config["data"]["batch_size"]), train_t, attn_type)
        step_ms = (row.get("stage_b") or {}).get("mean_step_ms")
        if step_ms and row["latency_ms"]:
            row["tokens_per_sec_train"] = (train_t * bs) / (step_ms / 1000.0)

        if variant == "dense_flash":
            full_steps = _resolve_train_steps(config, "dense_pretrain")
        else:
            full_steps = _resolve_train_steps(config, "finetune_from_dense")
        if step_ms:
            row["est_full_train_min"] = step_ms * full_steps / 60000.0

        del model
        reset_peak_vram(device)
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "oom"
        row["error"] = str(exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()

    row["profile_wall_sec"] = time.perf_counter() - t_total
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fast", "full"), default="full")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--context-lens", type=int, nargs="*")
    parser.add_argument("--dense-smoke-steps", type=int, default=2)
    parser.add_argument("--sparse-smoke-steps", type=int, default=2)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    smoke_override = {
        "transformer": {
            "dense_pretrain_steps": args.dense_smoke_steps,
            "sparse_finetune_steps": args.sparse_smoke_steps,
        }
    }
    config = _profile_config(args.profile)
    config_for_smoke = merge_configs(config, smoke_override)
    device = init_experiment_runtime(config_for_smoke)
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    context_lengths = args.context_lens or bench_cfg.context_lengths

    print("=== Experiment 7 three-stage profiler ===")
    print("Backends:", backend_status())
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"WARNING: {exc}")

    n_layers = config.get("model", {}).get("n_layers")
    dense_full = _resolve_train_steps(config, "dense_pretrain")
    sparse_full = _resolve_train_steps(config, "finetune_from_dense")
    router_idx_steps = config.get("index_pretrain", {}).get("router_max_steps", "?")

    print(f"profile={args.profile}  n_layers={n_layers}  T={context_lengths}")
    print(f"full steps: dense={dense_full}  sparse={sparse_full}  router_index={router_idx_steps}")
    print(f"smoke: dense={args.dense_smoke_steps}  sparse={args.sparse_smoke_steps}")
    print()

    with tempfile.TemporaryDirectory(prefix="exp7_profile_") as tmp:
        ckpt_dir = Path(tmp) / "dense"
        index_dir = Path(tmp) / "index"
        ckpt_dir.mkdir()
        index_dir.mkdir()
        rows: list[dict] = []

        for train_t in sorted(context_lengths, reverse=True):
            print(f"--- T={train_t} ---")
            for variant in args.variants:
                row = profile_cell(
                    config_for_smoke,
                    args.profile,
                    variant,
                    train_t,
                    device,
                    ckpt_dir,
                    index_dir,
                    dense_smoke_steps=args.dense_smoke_steps,
                    sparse_smoke_steps=args.sparse_smoke_steps,
                    index_smoke=True,
                )
                rows.append(row)
                audit = (row.get("training_audit") or {}).get("checks_passed")
                step_ms = (row.get("stage_b") or row.get("stage_a") or {}).get("mean_step_ms")
                addr_train = (row.get("routing_setup") or {}).get("address_params_trainable")
                router_frozen = (row.get("routing_setup") or {}).get("router_params_trainable")
                print(
                    f"  {variant:22s} {row['status']:5s}  "
                    f"step={step_ms:6.0f}ms" if step_ms else f"  {variant:22s} {row['status']:5s}  step=   n/a",
                    f"lat={row.get('latency_ms') or 0:6.1f}ms",
                    f"vram={row.get('peak_vram_mb') or 0:5.0f}MB",
                    f"audit={'OK' if audit else 'FAIL'}",
                    f"addr_train={addr_train}" if addr_train is not None else "",
                    f"router_frozen={router_frozen}" if router_frozen is not None else "",
                    sep="  ",
                )
                if row.get("error"):
                    print(f"      {row['error'][:120]}")

        # Wall-time estimates
        n_t = len(context_lengths)
        n_var = len(args.variants)
        est_dense = sum(
            r.get("est_full_train_min", 0) or 0
            for r in rows if r["variant"] == "dense_flash" and r["status"] == "ok"
        )
        est_sparse = sum(
            r.get("est_full_train_min", 0) or 0
            for r in rows if r["variant"] != "dense_flash" and r["status"] == "ok"
        )
        # Stage A once per T (not per variant row duplicate — dense rows are per-T)
        stage_a_unique = est_dense  # one dense row per T
        index_est_min = n_t * float(router_idx_steps or 0) * 0.05  # rough placeholder

        print()
        print("=== Full-suite time estimates ===")
        print(f"  Stage A dense ({n_t} T):           ~{stage_a_unique:.0f} min")
        print(f"  Stage A.5 router index ({n_t} T):   ~{index_est_min:.0f} min (rough)")
        print(f"  Stage B variants ({n_t}×{n_var-1}):  ~{est_sparse:.0f} min")
        print(f"  Total train (rough):                 ~{stage_a_unique + est_sparse + index_est_min:.0f} min")

        n_ok = sum(1 for r in rows if r["status"] == "ok")
        n_oom = sum(1 for r in rows if r["status"] == "oom")
        n_err = sum(1 for r in rows if r["status"] == "error")
        print(f"\nSmoke: ok={n_ok}  oom={n_oom}  error={n_err}")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "n_layers": n_layers,
            "context_lengths": context_lengths,
            "protocol": "three_stage_dense_index_sparse",
            "estimates_min": {
                "stage_a_dense": stage_a_unique,
                "stage_b_variants": est_sparse,
                "stage_a5_router_index_rough": index_est_min,
            },
            "rows": rows,
            "summary": {"ok": n_ok, "oom": n_oom, "error": n_err},
        }
        out = Path(args.output) if args.output else (
            ROOT / "experiments" / "Experiment_7" / "suite_long_context"
            / f"profile_{args.profile}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {out}")
        if n_err > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
