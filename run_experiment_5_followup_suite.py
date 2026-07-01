#!/usr/bin/env python3
"""
Experiment 5 follow-up suite — post Exp-4 fair baseline & characterization.

Ordered by progress-efficiency (highest expected learning per GPU-hour first):

  1. step0_eval           — all 4 variants, 0 fine-tune (init quality)
  2. dense_rescue_4k      — dense @ 4k steps (healthy region from Exp 4 logs)
  3. dense_rescue_lr1e4   — dense @ 10k steps, lr=1e-4
  4. dense_best_ckpt      — dense @ 10k, eval every 500, eval best checkpoint
  5. fair_compare_4k      — all 4 variants @ 4k matched steps
  6. collapse_dense_s42   — dense @ 10k seed=42 (collapse reproducibility)
  7. collapse_dense_s123  — dense @ 10k seed=123
  8. collapse_keyvec_s42  — key_vector @ 10k seed=42
  9. collapse_keyvec_s123 — key_vector @ 10k seed=123
 10. scaling_benchmark    — forward-only seq-len sweep (reuse Exp 4 checkpoints)

Usage:
  python run_experiment_5_followup_suite.py --dry-run
  python run_experiment_5_followup_suite.py
  python run_experiment_5_followup_suite.py --start 1 --end 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments import experiment_4
from experiments.common import resolve_reuse_transformer_checkpoint
from routing_attention.utils.config import load_config, merge_configs
from routing_attention.utils.experiment import get_experiments_root, resolve_checkpoint_path

TRANSFORMER_CKPT_REL = "Experiment_1/run_025/checkpoints/transformer/final.pt"
ROUTER_ASYMMETRIC_CKPT_REL = "Experiment_1/run_052/checkpoints/router/best.pt"
LEARNED_ADDRESS_CKPT_REL = "Experiment_1/run_051/checkpoints/addresses/best.pt"

# Exp 4 full-run checkpoints for scaling benchmark (forward-only)
EXP4_CHECKPOINTS = {
    "routing_asymmetric": "Experiment_4/run_023/checkpoints/routing_asymmetric_final.pt",
    "learned_address_k32": "Experiment_4/run_024/checkpoints/learned_address_k32_final.pt",
    "key_vector_k32": "Experiment_4/run_025/checkpoints/key_vector_k32_final.pt",
    "dense": "Experiment_4/run_026/checkpoints/dense_final.pt",
}

FOUR_VARIANTS = [
    "routing_asymmetric",
    "learned_address_k32",
    "key_vector_k32",
    "dense",
]

# Longest first: stress-test OOM / shape bugs early; crossover data arrives sooner.
FULL_SEQ_LENS = [16384, 8192, 4096, 2048, 784]
DRY_SEQ_LENS = [2048, 784]

SUITE_RUNS: list[dict] = [
    {
        "suite_index": 1,
        "run_id": "step0_eval",
        "mode": "multi",
        "variants": FOUR_VARIANTS,
        "hypothesis": "Pre-finetune quality is already high; fine-tune may help sparse but hurt dense.",
        "changes": "Load run_025/final.pt, 0 fine-tune steps, eval LM + digit acc for all 4 variants.",
        "why": "Cheapest run — establishes whether 10k fine-tune is even necessary.",
        "dry_max_steps": 0,
        "full_max_steps": 0,
        "skip_finetune": True,
    },
    {
        "suite_index": 2,
        "run_id": "dense_rescue_4k",
        "mode": "single",
        "variant": "dense",
        "hypothesis": "Dense stays healthy at 4k steps (before Exp-4 collapse zone ~5k+).",
        "changes": "dense, max_steps=4000, lr=3e-4, same init.",
        "why": "Establish fair dense ceiling without catastrophic divergence.",
        "dry_max_steps": 40,
        "full_max_steps": 4000,
    },
    {
        "suite_index": 3,
        "run_id": "dense_rescue_lr1e4",
        "mode": "single",
        "variant": "dense",
        "hypothesis": "Lower LR prevents dense fine-tune collapse at 10k steps.",
        "changes": "dense, max_steps=10000, lr=1e-4.",
        "why": "Tests if collapse is optimization-driven (LR too high).",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "lr": 1e-4,
    },
    {
        "suite_index": 4,
        "run_id": "dense_best_ckpt",
        "mode": "single",
        "variant": "dense",
        "hypothesis": "Best eval checkpoint during 10k training beats final collapsed weights.",
        "changes": "dense, max_steps=10000, eval_every=500, evaluate best.pt not final.",
        "why": "Fair dense baseline with early-stopping-style checkpoint selection.",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "eval_every": 500,
        "use_best_checkpoint": True,
    },
    {
        "suite_index": 5,
        "run_id": "fair_compare_4k",
        "mode": "multi",
        "variants": FOUR_VARIANTS,
        "hypothesis": "At matched 4k steps, sparse variants are within epsilon of healthy dense.",
        "changes": "All 4 variants, max_steps=4000, lr=3e-4, frozen router/addresses.",
        "why": "The head-to-head comparison Exp-4 intended but at safe step count.",
        "dry_max_steps": 40,
        "full_max_steps": 4000,
    },
    {
        "suite_index": 6,
        "run_id": "collapse_dense_s42",
        "mode": "single",
        "variant": "dense",
        "hypothesis": "Dense collapse at 10k/3e-4 is reproducible (seed 42).",
        "changes": "dense, max_steps=10000, lr=3e-4, seed=42.",
        "why": "Confirm regularization story is systematic not fluke.",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "seed": 42,
    },
    {
        "suite_index": 7,
        "run_id": "collapse_dense_s123",
        "mode": "single",
        "variant": "dense",
        "hypothesis": "Dense collapse reproduces across seeds.",
        "changes": "dense, max_steps=10000, lr=3e-4, seed=123.",
        "why": "Second seed for collapse characterization.",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "seed": 123,
    },
    {
        "suite_index": 8,
        "run_id": "collapse_keyvec_s42",
        "mode": "single",
        "variant": "key_vector_k32",
        "hypothesis": "Key-vector sparse stays stable at 10k/3e-4 (seed 42).",
        "changes": "key_vector_k32, max_steps=10000, lr=3e-4, seed=42.",
        "why": "Paired with dense collapse — sparse regularization control.",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "seed": 42,
    },
    {
        "suite_index": 9,
        "run_id": "collapse_keyvec_s123",
        "mode": "single",
        "variant": "key_vector_k32",
        "hypothesis": "Key-vector stability holds across seeds.",
        "changes": "key_vector_k32, max_steps=10000, lr=3e-4, seed=123.",
        "why": "Second seed for sparse stability.",
        "dry_max_steps": 40,
        "full_max_steps": 10000,
        "seed": 123,
    },
    {
        "suite_index": 10,
        "run_id": "scaling_benchmark",
        "mode": "scaling",
        "variants": FOUR_VARIANTS,
        "hypothesis": "Sparse forward-pass advantage grows with sequence length.",
        "changes": "Forward-only latency sweep; load Exp-4 final checkpoints when available.",
        "why": "Efficiency breakthrough narrative without extra training.",
        "dry_seq_lens": DRY_SEQ_LENS,
        "full_seq_lens": FULL_SEQ_LENS,
        "skip_finetune": True,
    },
]


def _max_steps(run: dict, dry_run: bool) -> int:
    if run["mode"] == "scaling":
        return 0
    return run["dry_max_steps"] if dry_run else run.get("full_max_steps", 10000)


def _build_config_override(run: dict, dry_run: bool) -> dict:
    max_steps = _max_steps(run, dry_run)
    override: dict = {
        "experiment": {
            "name": "Experiment_5",
            "description": f"Follow-up: {run['run_id']}",
        },
        "idea_manifest": {
            "suite_name": "experiment_5_followup",
            "suite_index": run["suite_index"],
            "run_id": run["run_id"],
            "mode": run["mode"],
            "hypothesis": run["hypothesis"],
            "changes": run["changes"],
            "why": run["why"],
            "dry_run": dry_run,
            "max_steps": max_steps,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "reuse": {
            "transformer_checkpoint": TRANSFORMER_CKPT_REL,
        },
        "routing_attention": {
            "max_steps": max_steps,
            "lr": run.get("lr", 3e-4),
        },
    }
    if "eval_every" in run:
        override["routing_attention"]["eval_every"] = run["eval_every"]
        override["validation"] = {"enabled": True, "max_batches": 3 if dry_run else 5}
    if "seed" in run:
        override["seed"] = run["seed"]
    if dry_run:
        override["evaluation"] = {"max_batches": 3}
    return override


def _preflight(scaling_in_suite: bool = False) -> list[str]:
    warnings: list[str] = []
    if scaling_in_suite:
        try:
            import faiss  # noqa: F401

            warnings.append(
                "FAISS available: scaling will use faiss_flat for T>2048 (optimized vector search)."
            )
        except ImportError:
            warnings.append(
                "FAISS not installed: scaling falls back to brute_force GEMM for all T>2048. "
                "Install faiss-gpu for optimized long-seq retrieval."
            )
        try:
            from routing_attention.retrieval.index import numpy_bridge_available
            import torch

            if torch.cuda.is_available():
                warnings.append(
                    f"CUDA device: {torch.cuda.get_device_name(0)} (forward pass runs on GPU)."
                )
            else:
                warnings.append("CUDA not available — benchmarks will run on CPU.")
            if not numpy_bridge_available():
                warnings.append(
                    "torch-to-numpy bridge broken: FAISS uses ctypes fallback (still works on GPU). "
                    "Keep numpy==1.26.4 for faiss-gpu; do not upgrade to numpy 2.x."
                )
        except Exception:
            pass
    base = load_config(ROOT / "configs" / "base.yaml")
    exp_cfg = load_config(ROOT / "configs" / "experiment_4.yaml")
    config = merge_configs(base, exp_cfg)
    if resolve_reuse_transformer_checkpoint(config) is None:
        warnings.append(f"Missing transformer checkpoint: {TRANSFORMER_CKPT_REL}")
    if resolve_checkpoint_path(ROUTER_ASYMMETRIC_CKPT_REL) is None:
        warnings.append(f"Missing router checkpoint: {ROUTER_ASYMMETRIC_CKPT_REL}")
    if resolve_checkpoint_path(LEARNED_ADDRESS_CKPT_REL) is None:
        warnings.append(f"Missing address checkpoint: {LEARNED_ADDRESS_CKPT_REL}")
    for var, rel in EXP4_CHECKPOINTS.items():
        if resolve_checkpoint_path(rel) is None:
            warnings.append(
                f"Scaling will use init weights for {var} (missing {rel})"
            )
    return warnings


def _extract_metrics(summary: dict) -> dict:
    """Pull primary metrics from single- or multi-variant summary."""
    lm_results = summary.get("lm_results") or {}
    if not lm_results:
        return {}
    if len(lm_results) == 1:
        key = next(iter(lm_results))
        m = lm_results[key]
        return {
            "primary_variant": key,
            "lm_loss": m.get("loss"),
            "perplexity": m.get("perplexity"),
            "digit_accuracy": m.get("digit_accuracy"),
        }
    return {
        "variant_metrics": {
            k: {"lm_loss": m.get("loss"), "digit_accuracy": m.get("digit_accuracy")}
            for k, m in lm_results.items()
        }
    }


def _invoke_run(run: dict, dry_run: bool) -> dict:
    override = _build_config_override(run, dry_run)
    skip_finetune = run.get("skip_finetune", False) or _max_steps(run, dry_run) <= 0
    use_best = run.get("use_best_checkpoint", False)

    if run["mode"] == "scaling":
        seq_lens = run["dry_seq_lens"] if dry_run else run["full_seq_lens"]
        max_seq = max(seq_lens)
        override["model"] = {"max_seq_len": max_seq}
        # All sparse variants use RoutingRetriever for scaling (key-vector included).
        override["retrieval"] = {
            "method": "auto",
            "apply_to_key_vector": True,
            "max_seq_len": max_seq,
            "dtype": "float16",
            "use_gpu": True,
        }
        # Long-seq forward benchmarks are memory-heavy; batch=1 keeps T=16k feasible.
        override["data"] = {"batch_size": 1, "num_workers": 0}
        override["evaluation"] = {
            **override.get("evaluation", {}),
            "max_batches": 1,
            "benchmark_runs": 2 if dry_run else 10,
            "benchmark_warmup": 1 if dry_run else 3,
        }
        all_metrics: dict = {}
        last_summary: dict = {}
        for variant in run["variants"]:
            ckpt_rel = EXP4_CHECKPOINTS.get(variant)
            ckpt_path = resolve_checkpoint_path(ckpt_rel) if ckpt_rel else None
            last_summary = experiment_4.run(
                variant=variant,
                dry_run=False,
                config_override=override,
                compare_all=False,
                skip_finetune=True,
                benchmark_seq_lens=seq_lens,
                load_checkpoint_path=str(ckpt_path) if ckpt_path else None,
            )
            bench = (last_summary.get("benchmark_results") or {}).get(variant, {})
            sweep = bench.get("seq_len_sweep", {})
            all_metrics[variant] = {
                str(sl): sweep.get(str(sl), sweep.get(sl, {})).get("latency_ms")
                for sl in seq_lens
            }
        return {
            **last_summary,
            "run_id": run["run_id"],
            "scaling_latency_ms": all_metrics,
            "lm_results": {},
            "lm_loss": None,
            "digit_accuracy": None,
            "verdict": "scaling",
        }

    if run["mode"] == "multi":
        return experiment_4.run(
            dry_run=False,
            config_override=override,
            compare_all=False,
            variants=run["variants"],
            skip_finetune=skip_finetune,
            use_best_checkpoint=use_best,
        )

    return experiment_4.run(
        variant=run["variant"],
        dry_run=False,
        config_override=override,
        compare_all=False,
        skip_finetune=skip_finetune,
        use_best_checkpoint=use_best,
    )


def _suite_summary(results: list[dict]) -> dict:
    """Build cross-run comparison table."""
    by_id = {r["run_id"]: r for r in results if r.get("status") == "ok"}

    step0 = by_id.get("step0_eval", {}).get("variant_metrics", {})
    fair4k = by_id.get("fair_compare_4k", {}).get("variant_metrics", {})
    dense_4k = by_id.get("dense_rescue_4k", {})
    dense_best = by_id.get("dense_best_ckpt", {})
    scaling = by_id.get("scaling_benchmark", {}).get("scaling_latency_ms", {})

    collapse = {
        "dense": {
            "s42": by_id.get("collapse_dense_s42", {}),
            "s123": by_id.get("collapse_dense_s123", {}),
        },
        "key_vector_k32": {
            "s42": by_id.get("collapse_keyvec_s42", {}),
            "s123": by_id.get("collapse_keyvec_s123", {}),
        },
    }

    return {
        "step0_eval": step0,
        "dense_rescue": {
            "dense_4k": {
                "lm_loss": dense_4k.get("lm_loss"),
                "digit_accuracy": dense_4k.get("digit_accuracy"),
            },
            "dense_best_ckpt": {
                "lm_loss": dense_best.get("lm_loss"),
                "digit_accuracy": dense_best.get("digit_accuracy"),
            },
            "dense_lr1e4": {
                "lm_loss": by_id.get("dense_rescue_lr1e4", {}).get("lm_loss"),
                "digit_accuracy": by_id.get("dense_rescue_lr1e4", {}).get("digit_accuracy"),
            },
        },
        "fair_compare_4k": fair4k,
        "collapse_repro": collapse,
        "scaling_latency_ms": scaling,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 5 follow-up suite")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with reduced steps/eval")
    parser.add_argument("--start", type=int, default=1, help="First suite_index (inclusive)")
    parser.add_argument("--end", type=int, default=len(SUITE_RUNS), help="Last suite_index (inclusive)")
    args = parser.parse_args()

    runs = [r for r in SUITE_RUNS if args.start <= r["suite_index"] <= args.end]
    if not runs:
        print(f"No runs in range {args.start}–{args.end}")
        return 1

    suite_log_dir = get_experiments_root() / "Experiment_5" / "suite_followup"
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_log_path = suite_log_dir / f"suite_run_{suite_stamp}.jsonl"

    print(f"Experiment 5 follow-up suite: {len(runs)} runs")
    print(f"Suite log: {suite_log_path}")
    if args.dry_run:
        print("DRY RUN — reduced steps / eval batches (step0 stays at 0 steps)")

    scaling_planned = any(r["run_id"] == "scaling_benchmark" for r in runs)
    for w in _preflight(scaling_in_suite=scaling_planned):
        print(f"WARNING: {w}")

    results: list[dict] = []
    for run in runs:
        label = f"[{run['suite_index']}/{len(SUITE_RUNS)}] {run['run_id']}"
        print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
        entry = {
            "suite_index": run["suite_index"],
            "run_id": run["run_id"],
            "mode": run["mode"],
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            summary = _invoke_run(run, dry_run=args.dry_run)
            metrics = _extract_metrics(summary)
            entry.update({
                "status": "ok",
                "run_dir": summary.get("run_dir"),
                "verdict": summary.get("verdict"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **metrics,
            })
            if summary.get("scaling_latency_ms"):
                entry["scaling_latency_ms"] = summary["scaling_latency_ms"]
            if entry.get("lm_loss") is not None:
                print(
                    f"OK {label} → lm_loss={entry.get('lm_loss')}, "
                    f"digit_acc={entry.get('digit_accuracy')}"
                )
            elif entry.get("variant_metrics"):
                print(f"OK {label} → {entry['variant_metrics']}")
            elif entry.get("scaling_latency_ms"):
                print(f"OK {label} → scaling sweep done")
            else:
                print(f"OK {label}")
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__}"
            entry.update({
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"FAILED {label}: {error_msg}")
            fail_dir = suite_log_dir / f"failed_{run['run_id']}_{suite_stamp}"
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "failure.txt").write_text(
                f"run_id: {run['run_id']}\n"
                f"hypothesis: {run['hypothesis']}\n"
                f"error: {error_msg}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            entry["failure_log"] = str(fail_dir / "failure.txt")

        results.append(entry)
        with open(suite_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    summary_path = suite_log_dir / f"suite_summary_{suite_stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "suite": "experiment_5_followup",
            "dry_run": args.dry_run,
            "runs": results,
            "analysis": _suite_summary(results),
        }, f, indent=2, default=str)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nSuite complete: {ok} ok, {failed} failed")
    print(f"Summary: {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
