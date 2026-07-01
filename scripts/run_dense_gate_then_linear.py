#!/usr/bin/env python3
"""Train dense @ T=2048; run linear only if dense official holdout >= gate threshold."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.gap_calibration import (  # noqa: E402
    extract_accuracy,
)


def run_arena(
    *,
    config: Path,
    output_dir: Path,
    variants: list[str],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "run_routing_arena_suite.py"),
        "--config",
        str(config),
        "--variants",
        *variants,
        "--output-dir",
        str(output_dir),
        "--skip-index-pretrain",
    ]
    log_path = output_dir / "run.log"
    print(f"=== Running variants={variants} ===")
    print(f"  log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n=== variants={variants} ===\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        )
    return proc.returncode


def read_dense_official(output_dir: Path) -> float | None:
    latest = output_dir / "latest.json"
    if not latest.exists():
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    dense = (data.get("results") or {}).get("dense_flash") or {}
    return extract_accuracy(dense, mid=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "routing_gap_decoys_scatter_t2048_4decoys.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "Experiment_7" / "gap_decoys_scatter_t2048_2L_4decoys",
    )
    parser.add_argument("--gate-min", type=float, default=0.20)
    args = parser.parse_args()

    out = args.output_dir
    rc = run_arena(config=args.config, output_dir=out, variants=["dense_flash"])
    if rc != 0:
        print(f"dense_flash failed exit_code={rc}")
        sys.exit(rc)

    dense_acc = read_dense_official(out)
    if dense_acc is None:
        print("Could not read dense official accuracy from latest.json")
        sys.exit(1)

    pct = dense_acc * 100.0
    print(f"\n=== Dense gate ===")
    print(f"  official holdout: {pct:.2f}% ({dense_acc:.4f})")
    print(f"  threshold: {args.gate_min * 100:.0f}%")

    summary = {
        "dense_official_acc": dense_acc,
        "gate_min": args.gate_min,
        "linear_ran": False,
        "linear_official_acc": None,
    }

    if dense_acc < args.gate_min:
        print(f"  SKIP linear — dense below {args.gate_min * 100:.0f}% gate")
        (out / "gate_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        sys.exit(0)

    print("  PASS — running linear on same task")
    rc = run_arena(config=args.config, output_dir=out, variants=["linear"])
    linear_acc = read_dense_official(out)  # wrong - need linear
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    linear_payload = (latest.get("results") or {}).get("linear") or {}
    linear_acc = extract_accuracy(linear_payload, mid=False)

    summary["linear_ran"] = True
    summary["linear_official_acc"] = linear_acc
    if linear_acc is not None and dense_acc is not None:
        summary["gap_pp"] = (dense_acc - linear_acc) * 100.0
    (out / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Final ===")
    print(f"  dense:  {pct:.2f}%")
    if linear_acc is not None:
        print(f"  linear: {linear_acc * 100:.2f}%")
        print(f"  gap:    {(dense_acc - linear_acc) * 100:.1f} pp")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
