#!/usr/bin/env python3
"""
Experiment 6 — fused Triton kernels scaling suite.

Compares Layer 2 (fused_causal retrieval only) vs Layer 3 (fused sparse attention
end-to-end) against dense baseline. Longest-first sequence sweep reuses Exp 4
checkpoints.

Runs:
  1. fused_causal_scaling  — RoutingRetriever fused_causal; separate meat path
  2. fused_sparse_scaling  — Layer 3: causal_topk + Triton sparse meat (all sparse variants + dense)

Usage:
  python run_experiment_6_fused_suite.py --dry-run
  python run_experiment_6_fused_suite.py
  python run_experiment_6_fused_suite.py --start 1 --end 2
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
from routing_attention.kernels.causal_topk import causal_topk_available
from routing_attention.utils.config import load_config, merge_configs
from routing_attention.utils.experiment import get_experiments_root, resolve_checkpoint_path

EXP4_CHECKPOINTS = {
    "routing_asymmetric": "Experiment_4/run_023/checkpoints/routing_asymmetric_final.pt",
    "learned_address_k32": "Experiment_4/run_024/checkpoints/learned_address_k32_final.pt",
    "key_vector_k32": "Experiment_4/run_025/checkpoints/key_vector_k32_final.pt",
    "dense": "Experiment_4/run_026/checkpoints/dense_final.pt",
    "dense_flash": "Experiment_4/run_026/checkpoints/dense_final.pt",
}

FOUR_VARIANTS = [
    "routing_asymmetric",
    "learned_address_k32",
    "key_vector_k32",
    "dense",
    "dense_flash",
]

FULL_SEQ_LENS = [32768, 16384, 8192, 4096, 2048, 784]
DRY_SEQ_LENS = [2048, 784]

SUITE_RUNS: list[dict] = [
    {
        "suite_index": 1,
        "run_id": "fused_causal_scaling",
        "mode": "scaling",
        "variants": FOUR_VARIANTS,
        "hypothesis": "Layer-2 fused causal top-K removes FAISS overhead at long T.",
        "changes": "method=fused_causal, apply_to_key_vector=true, use_fused_sparse=false.",
        "why": "Isolates retrieval kernel speedup without changing meat attention.",
        "retrieval": {
            "method": "fused_causal",
            "apply_to_key_vector": True,
            "use_fused_sparse": False,
            "dtype": "float16",
            "use_gpu": True,
        },
    },
    {
        "suite_index": 2,
        "run_id": "fused_sparse_scaling",
        "mode": "scaling",
        "variants": FOUR_VARIANTS,
        "hypothesis": "Layer-3 fused sparse attention crosses below dense at long T.",
        "changes": "method=fused_causal, use_fused_sparse=true (full Triton pipeline).",
        "why": "End-to-end sparse path — retrieval + gather + softmax in fused kernels.",
        "retrieval": {
            "method": "fused_causal",
            "apply_to_key_vector": True,
            "use_fused_sparse": True,
            "dtype": "float16",
            "use_gpu": True,
        },
    },
]


def _preflight() -> list[str]:
    warnings: list[str] = []
    import torch

    if not torch.cuda.is_available():
        warnings.append("CUDA not available — fused kernels will fall back to PyTorch.")
    elif not causal_topk_available():
        warnings.append("Triton not available — fused_causal will use streaming fallback.")
    else:
        warnings.append(f"Fused causal top-K available on {torch.cuda.get_device_name()}.")
    return warnings


def _build_config_override(run: dict, dry_run: bool, seq_lens: list[int]) -> dict:
    max_seq = max(seq_lens)
    base = load_config(ROOT / "configs" / "base.yaml")
    exp6 = load_config(ROOT / "configs" / "experiment_6.yaml")
    override = merge_configs(base, exp6)
    override["experiment"] = {
        "name": "Experiment_6",
        "description": f"Fused kernels: {run['run_id']}",
    }
    override["idea_manifest"] = {
        "run_id": run["run_id"],
        "mode": run["mode"],
        "hypothesis": run.get("hypothesis", ""),
        "changes": run.get("changes", ""),
    }
    override["model"] = {"max_seq_len": max_seq}
    override["retrieval"] = {
        **override.get("retrieval", {}),
        **run["retrieval"],
        "max_seq_len": max_seq,
    }
    override["data"] = {"batch_size": 1, "num_workers": 0}
    override["evaluation"] = {
        **override.get("evaluation", {}),
        "max_batches": 1,
        "benchmark_runs": 2 if dry_run else 10,
        "benchmark_warmup": 1 if dry_run else 3,
    }
    return override


def _invoke_run(run: dict, dry_run: bool) -> dict:
    seq_lens = DRY_SEQ_LENS if dry_run else FULL_SEQ_LENS
    override = _build_config_override(run, dry_run, seq_lens)
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
        "verdict": "fused_scaling",
        "use_fused_sparse": run["retrieval"].get("use_fused_sparse", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6 fused kernel scaling suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=len(SUITE_RUNS))
    args = parser.parse_args()

    runs = [r for r in SUITE_RUNS if args.start <= r["suite_index"] <= args.end]
    print(f"Experiment 6 fused suite: runs {args.start}–{args.end} ({len(runs)} planned)")
    for w in _preflight():
        print(f"  preflight: {w}")

    suite_dir = get_experiments_root() / "Experiment_6" / "suite_fused"
    suite_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = suite_dir / f"suite_run_{stamp}.jsonl"
    summary_entries: list[dict] = []

    for run in runs:
        label = f"[{run['suite_index']}] {run['run_id']}"
        print(f"\n=== {label} ===")
        entry: dict = {
            "suite_index": run["suite_index"],
            "run_id": run["run_id"],
            "mode": run["mode"],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            summary = _invoke_run(run, args.dry_run)
            entry["status"] = "ok"
            entry["scaling_latency_ms"] = summary.get("scaling_latency_ms", {})
            entry["use_fused_sparse"] = summary.get("use_fused_sparse", False)
            entry["verdict"] = summary.get("verdict", "fused_scaling")
            print(f"OK {label}")
            for variant, latencies in entry.get("scaling_latency_ms", {}).items():
                for sl, ms in latencies.items():
                    if ms is not None:
                        print(f"  {variant} T={sl}: {ms:.2f} ms")
        except Exception:
            entry["status"] = "error"
            entry["traceback"] = traceback.format_exc()
            print(f"FAIL {label}\n{entry['traceback']}")
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        summary_entries.append(entry)

    summary_path = suite_dir / f"suite_summary_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "Experiment_6",
                "dry_run": args.dry_run,
                "runs": summary_entries,
            },
            f,
            indent=2,
        )
    print(f"\nSuite summary: {summary_path}")


if __name__ == "__main__":
    main()
