#!/usr/bin/env python3
"""Sequential capacity sweep for massive_addr_val @ T=2048 (dense_flash).

Runs one full train+eval arena job per ``num_kv_pairs`` value, each in its own
output folder so runs are easy to tell apart.

Default sweep: N ∈ {2, 5, 7, 10, 25, 50}
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    resolve_dense_checkpoint,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

DEFAULT_SWEEP = [2, 5, 7, 10, 25, 50]
DEFAULT_BASE_CONFIG = ROOT / "configs" / "routing_arena_massive_addr_val_t2048.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "massive_addr_val_capacity_sweep"


class _Tee:
    """Mirror stdout/stderr to a log file."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file = log_path.open("a", encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== Capacity sweep preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available.")
    if not info.get("fla_linear"):
        print("ERROR: flash-linear-attention required.")
        sys.exit(1)
    assert_production_backends_available()
    print()
    return info


def _patch_arena_for_n(arena_cfg: dict, num_kv_pairs: int) -> dict:
    cfg = copy.deepcopy(arena_cfg)
    bench = dict(cfg.get("long_context_benchmark", {}))
    bench["task_types"] = ["massive_addr_val"]
    bench["num_kv_pairs"] = int(num_kv_pairs)
    bench["num_distractors"] = 0
    cfg["long_context_benchmark"] = bench
    cfg["description"] = (
        f"massive_addr_val capacity sweep @ T=2048 — N={num_kv_pairs} KV pairs (dense_flash)"
    )
    cfg["variants"] = ["dense_flash"]
    return cfg


def _extract_metrics(payload: dict) -> dict:
    ev = payload.get("eval_official") or payload.get("eval", {})
    train_info = payload.get("train_info", {})
    best = train_info.get("best_holdout", {})
    return {
        "official_gate_accuracy": float(
            ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0.0)) or 0.0
        ),
        "official_gate_correct": int(ev.get("primary_gate_correct", ev.get("correct", 0)) or 0),
        "official_gate_total": int(ev.get("primary_gate_total", ev.get("total", 0)) or 0),
        "best_holdout_step": best.get("step"),
        "best_holdout_accuracy": float(best.get("primary_gate_accuracy", best.get("overall_accuracy", 0.0)) or 0.0),
        "trained_steps": train_info.get("trained_steps"),
        "final_loss": train_info.get("final_loss"),
        "by_task_type": ev.get("by_task_type", {}),
    }


def _run_one(
    *,
    num_kv_pairs: int,
    arena_cfg: dict,
    dry_run: bool,
    n_layers: int | None,
    dense_checkpoint: Path | None,
    output_root: Path,
    preflight_info: dict,
) -> dict:
    run_dir = output_root / f"N{num_kv_pairs:03d}_massive_addr_val_dense_flash"
    run_dir.mkdir(parents=True, exist_ok=True)

    patched_arena = _patch_arena_for_n(arena_cfg, num_kv_pairs)
    train_t = int(patched_arena["train_context_length"])
    layers = n_layers or int(patched_arena.get("n_layers", 6))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "massive_addr_val_capacity_sweep_run",
        "task_type": "massive_addr_val",
        "num_kv_pairs": num_kv_pairs,
        "train_context_length": train_t,
        "n_layers": layers,
        "variant": "dense_flash",
        "dry_run": dry_run,
        "description": patched_arena.get("description"),
        "output_dir": str(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_path = run_dir / "run.log"
    if log_path.exists():
        log_path.unlink()
    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = tee_out, tee_err

    status = "ok"
    payload: dict = {}
    err_text = ""
    try:
        print(f"\n{'=' * 72}")
        print(f"SWEEP RUN  num_kv_pairs={num_kv_pairs}  output={run_dir}")
        print(f"{'=' * 72}\n")

        config = build_arena_experiment_config(patched_arena, dry_run=dry_run, n_layers=layers)
        dense_ckpt = resolve_dense_checkpoint(
            train_t,
            n_layers=layers,
            explicit=dense_checkpoint or patched_arena.get("dense_checkpoint"),
        )
        print(f"  dense_checkpoint={dense_ckpt}")
        print(f"  holdout_total={config.get('holdout', {}).get('total_samples', 300)}")
        steps = int(
            config.get("transformer", {}).get("sparse_finetune_steps")
            or config.get("transformer", {}).get("max_steps", 0)
        )
        print(f"  training_steps={steps}")
        print()

        device = init_arena_runtime(config)
        log = logging.getLogger("routing_arena.train")
        reset_peak_vram(device)

        payload = run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            device=device,
            log=log,
        )

        metrics = _extract_metrics(payload)
        ev = payload.get("eval_official") or payload.get("eval", {})
        gate = metrics["official_gate_accuracy"]
        print(
            f"\nOK N={num_kv_pairs}: official_gate={gate * 100:.2f}% "
            f"({metrics['official_gate_correct']}/{metrics['official_gate_total']}) "
            f"best@{metrics['best_holdout_step']}={metrics['best_holdout_accuracy'] * 100:.2f}%"
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = "dry_run" if dry_run else "full"
        run_summary = {
            **manifest,
            "dense_checkpoint": str(dense_ckpt),
            "preflight": preflight_info,
            "metrics": metrics,
            "results": {"dense_flash": payload},
        }
        summary_path = run_dir / f"routing_arena_{tag}_{stamp}.json"
        summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        (run_dir / "latest.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "num_kv_pairs": num_kv_pairs,
            "status": status,
            "output_dir": str(run_dir),
            "metrics": metrics,
            "summary_path": str(summary_path),
        }
    except Exception:
        status = "error"
        err_text = traceback.format_exc()
        print(err_text)
        return {
            "num_kv_pairs": num_kv_pairs,
            "status": status,
            "output_dir": str(run_dir),
            "error": err_text,
        }
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        tee_out.close()
        tee_err.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="massive_addr_val KV capacity sweep (dense_flash)")
    parser.add_argument("--dry-run", action="store_true", help="Use dry_run profile from base config")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Arena YAML used as template (num_kv_pairs overridden per run)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder; each N gets N{NNN}_massive_addr_val_dense_flash/",
    )
    parser.add_argument(
        "--num-kv-pairs",
        type=int,
        nargs="+",
        default=None,
        help=f"Override sweep list (default: {DEFAULT_SWEEP})",
    )
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sweep_values = list(args.num_kv_pairs or DEFAULT_SWEEP)
    arena_cfg = load_routing_arena_config(args.base_config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    preflight_info = _preflight(args.dry_run)

    sweep_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "massive_addr_val_capacity_sweep",
        "task_type": "massive_addr_val",
        "train_context_length": int(arena_cfg["train_context_length"]),
        "variant": "dense_flash",
        "dry_run": args.dry_run,
        "num_kv_pairs_sweep": sweep_values,
        "base_config": str(args.base_config),
        "output_root": str(output_root),
    }
    (output_root / "sweep_manifest.json").write_text(json.dumps(sweep_meta, indent=2), encoding="utf-8")

    print("=== Capacity sweep plan ===")
    print(f"  values={sweep_values}")
    print(f"  output_root={output_root}")
    print(f"  base_config={args.base_config}")
    print()

    run_records: list[dict] = []
    for n in sweep_values:
        record = _run_one(
            num_kv_pairs=n,
            arena_cfg=arena_cfg,
            dry_run=args.dry_run,
            n_layers=args.n_layers,
            dense_checkpoint=args.dense_checkpoint,
            output_root=output_root,
            preflight_info=preflight_info,
        )
        run_records.append(record)

    table = []
    for rec in run_records:
        if rec["status"] == "ok":
            m = rec["metrics"]
            table.append(
                {
                    "num_kv_pairs": rec["num_kv_pairs"],
                    "official_gate_pct": round(m["official_gate_accuracy"] * 100, 2),
                    "correct": m["official_gate_correct"],
                    "total": m["official_gate_total"],
                    "best_step": m["best_holdout_step"],
                    "best_holdout_pct": round((m["best_holdout_accuracy"] or 0) * 100, 2),
                    "output_dir": rec["output_dir"],
                }
            )
        else:
            table.append(
                {
                    "num_kv_pairs": rec["num_kv_pairs"],
                    "status": "error",
                    "output_dir": rec["output_dir"],
                }
            )

    summary = {
        **sweep_meta,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runs": run_records,
        "results_table": table,
    }
    summary_path = output_root / "capacity_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("CAPACITY SWEEP SUMMARY")
    print(f"{'=' * 72}")
    print(f"  wrote: {summary_path}")
    for row in table:
        if row.get("status") == "error":
            print(f"  N={row['num_kv_pairs']:3d}  ERROR  -> {row['output_dir']}")
        else:
            print(
                f"  N={row['num_kv_pairs']:3d}  gate={row['official_gate_pct']:5.2f}% "
                f"({row['correct']}/{row['total']})  best@{row['best_step']}  "
                f"-> {row['output_dir']}"
            )

    failed = [r for r in run_records if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
