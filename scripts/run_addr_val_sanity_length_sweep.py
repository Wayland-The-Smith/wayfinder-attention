#!/usr/bin/env python3
"""Sequential addr_val sanity length sweep (Benchmark A) for dense_flash.

Runs the same sanity recipe at T in {512, 1024, 2048} with from-scratch dense training.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "configs" / "routing_arena_addr_val_sanity_t512.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "addr_val_sanity_length_sweep"
DEFAULT_LENGTHS = [512, 1024, 2048]
PYTHON = sys.executable


def _load_arena_cfg(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("routing_arena", raw)


def _patch_for_length(base: dict, train_t: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["train_context_length"] = train_t
    cfg["description"] = (
        f"addr_val sanity @ T={train_t} — 1-digit, 0 decoys, non-scattered, from-scratch dense"
    )
    bench = dict(cfg.get("long_context_benchmark", {}))
    bench["context_lengths"] = [train_t]
    cfg["long_context_benchmark"] = bench
    return cfg


def _write_temp_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"routing_arena": cfg}, sort_keys=False), encoding="utf-8")


def run_one(
    *,
    train_t: int,
    base_config: Path,
    output_root: Path,
    steps: int | None,
    dry_run: bool,
) -> dict:
    base = _load_arena_cfg(base_config)
    if steps is not None:
        base.setdefault("transformer", {})["sparse_finetune_steps"] = int(steps)
    cfg = _patch_for_length(base, train_t)
    run_dir = output_root / f"T{train_t}_addr_val_sanity_dense_flash"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = run_dir / "config_used.yaml"
    _write_temp_config(cfg, tmp_cfg)

    cmd = [
        PYTHON,
        str(ROOT / "run_routing_arena_suite.py"),
        "--config",
        str(tmp_cfg),
        "--variants",
        "dense_flash",
        "--output-dir",
        str(run_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")

    log_path = run_dir / "run.log"
    print(f"\n=== T={train_t} -> {run_dir} ===")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT)},
        )
    latest = run_dir / "latest.json"
    summary: dict = {
        "train_t": train_t,
        "run_dir": str(run_dir),
        "exit_code": proc.returncode,
        "config": str(tmp_cfg),
    }
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8"))
        dense = payload.get("results", {}).get("dense_flash", {})
        eval_block = dense.get("eval_official") or dense.get("eval") or {}
        summary["accuracy"] = eval_block.get("accuracy")
        summary["by_task"] = eval_block.get("by_task")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="addr_val sanity length sweep (dense_flash)")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--steps", type=int, default=None, help="Override sparse_finetune_steps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for train_t in args.lengths:
        results.append(
            run_one(
                train_t=train_t,
                base_config=args.base_config,
                output_root=args.output_root,
                steps=args.steps,
                dry_run=args.dry_run,
            )
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {
        "kind": "addr_val_sanity_length_sweep",
        "timestamp": stamp,
        "base_config": str(args.base_config),
        "lengths": args.lengths,
        "runs": results,
    }
    summary_path = args.output_root / f"sweep_summary_{stamp}.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.output_root / "latest_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    for row in results:
        acc = row.get("accuracy")
        acc_s = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "n/a"
        print(f"  T={row['train_t']}: exit={row['exit_code']} accuracy={acc_s}")


if __name__ == "__main__":
    main()
