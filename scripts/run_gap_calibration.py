#!/usr/bin/env python3
"""Progress-efficient dense vs linear gap calibration (Phases 1–3).

Phase 1: difficulty ladder @ fixed T with 3k-step screening (mid-holdout).
Phase 2: binary search on the best knob from Phase 1.
Phase 3: confirm on official 300-sample holdout (same step budget as screening).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.gap_calibration import (  # noqa: E402
    ADVANCE_GAP_MAX,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TEMPLATE,
    GAP_STOP_MIN,
    KnobSettings,
    append_csv_row,
    build_arena_config,
    gap_met,
    parse_run_results,
    parse_run_results_official,
    phase1_ladder,
    score_pair,
    write_config,
)

PYTHON = sys.executable
SCREEN_STEPS = 10000
CONFIRM_STEPS = 10000
DEFAULT_N_LAYERS = 2


def run_arena_pair(
    *,
    cfg: dict,
    run_dir: Path,
    variants: list[str],
    skip_index: bool = True,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_used.yaml"
    write_config(cfg, cfg_path)
    cmd = [
        PYTHON,
        str(ROOT / "run_routing_arena_suite.py"),
        "--config",
        str(cfg_path),
        "--variants",
        *variants,
        "--output-dir",
        str(run_dir),
    ]
    if skip_index:
        cmd.append("--skip-index-pretrain")
    log_path = run_dir / "run.log"
    print(f"  -> {run_dir.name}  variants={variants}")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        )
    return proc.returncode


def record_run(
    *,
    csv_path: Path,
    phase: str,
    run_id: str,
    knobs: KnobSettings,
    run_dir: Path,
    steps: int,
    exit_code: int,
    mid: bool = True,
) -> dict:
    latest = run_dir / "latest.json"
    accs = parse_run_results(latest) if mid else parse_run_results_official(latest)
    dense = accs.get("dense_flash")
    linear = accs.get("linear")
    local = accs.get("local_window64")
    gap = (dense - linear) if dense is not None and linear is not None else None
    row = {
        "phase": phase,
        "run_id": run_id,
        "label": knobs.label,
        "knobs_json": json.dumps(knobs.to_dict()),
        "steps": steps,
        "eval_mode": "mid" if mid else "official",
        "dense_acc": dense,
        "linear_acc": linear,
        "local_acc": local,
        "gap": gap,
        "score": score_pair(dense, linear),
        "exit_code": exit_code,
        "run_dir": str(run_dir),
    }
    append_csv_row(csv_path, row)
    gap_s = f"{gap * 100:.1f}pp" if gap is not None else "n/a"
    dense_s = f"{dense * 100:.1f}%" if dense is not None else "n/a"
    linear_s = f"{linear * 100:.1f}%" if linear is not None else "n/a"
    print(f"    dense={dense_s}  linear={linear_s}  gap={gap_s}  score={row['score']}")
    return row


def run_phase1(
    *,
    output_root: Path,
    train_t: int,
    steps: int,
    csv_path: Path,
) -> tuple[KnobSettings | None, dict | None]:
    print("\n=== Phase 1: difficulty ladder (mid-holdout screening) ===")
    results: list[dict] = []
    winner: KnobSettings | None = None
    winner_row: dict | None = None

    for run_id, knobs in phase1_ladder():
        run_dir = output_root / "phase1" / run_id
        cfg = build_arena_config(knobs, train_t=train_t, steps=steps)
        exit_code = run_arena_pair(
            cfg=cfg,
            run_dir=run_dir,
            variants=["dense_flash", "linear"],
        )
        row = record_run(
            phase="1",
            run_id=run_id,
            knobs=knobs,
            run_dir=run_dir,
            steps=steps,
            exit_code=exit_code,
            csv_path=csv_path,
        )
        results.append(row)
        if gap_met(row.get("dense_acc"), row.get("linear_acc")):
            winner = knobs
            winner_row = row
            print(f"  STOP Phase 1: gap target met at {run_id}")
            break

    if winner is None:
        eligible = [r for r in results if r.get("score") is not None]
        if eligible:
            winner_row = max(eligible, key=lambda r: r["score"])
            kd = json.loads(winner_row["knobs_json"])
            winner = KnobSettings(
                label=kd.get("label", winner_row["label"]),
                scatter_multi_needles=kd.get("scatter_multi_needles"),
                num_distractors=kd.get("num_distractors"),
                synthetic_decoy_addrs=kd.get("synthetic_decoy_addrs"),
                needle_depths=kd.get("needle_depths"),
                task_types=kd.get("task_types"),
                synthetic_conflict_rows=kd.get("synthetic_conflict_rows"),
            )
            print(f"  Phase 1 best (no stop): {winner_row['run_id']} score={winner_row['score']:.3f}")
        else:
            print("  Phase 1: no valid runs — falling back to L0 baseline")
            winner = KnobSettings("baseline")

    # L5: scatter + decoys if best had partial gap but not stop
    if winner_row and winner_row.get("gap") is not None:
        gap = winner_row["gap"]
        if ADVANCE_GAP_MAX <= gap < GAP_STOP_MIN:
            print("\n  Phase 1 L5: scatter + decoys_4 combined")
            l5 = KnobSettings(
                "scatter_decoys_4",
                scatter_multi_needles=True,
                num_distractors=4,
            )
            run_dir = output_root / "phase1" / "L5"
            cfg = build_arena_config(l5, train_t=train_t, steps=steps)
            exit_code = run_arena_pair(cfg=cfg, run_dir=run_dir, variants=["dense_flash", "linear"])
            row = record_run(
                phase="1",
                run_id="L5",
                knobs=l5,
                run_dir=run_dir,
                steps=steps,
                exit_code=exit_code,
                csv_path=csv_path,
            )
            if row.get("score") is not None and (
                winner_row.get("score") is None or row["score"] > winner_row["score"]
            ):
                winner = l5
                winner_row = row

    return winner, winner_row


def binary_search_values(knobs: KnobSettings) -> list[int]:
    kd = knobs.to_dict()
    if kd.get("num_distractors") is not None or knobs.num_distractors is not None:
        return [2, 3, 4, 5, 6, 7, 8]
    if kd.get("synthetic_conflict_rows") is not None or knobs.synthetic_conflict_rows is not None:
        return [2, 3, 4, 5]
    if kd.get("scatter_multi_needles"):
        return [0, 1]  # handled separately
    return [2, 4, 6, 8]


def run_phase2(
    *,
    output_root: Path,
    train_t: int,
    steps: int,
    csv_path: Path,
    seed_knobs: KnobSettings,
    seed_row: dict | None,
) -> tuple[KnobSettings, dict | None]:
    print("\n=== Phase 2: binary search on winning knob ===")
    kd = seed_knobs.to_dict()

    if seed_knobs.task_types and "addr_val_conflict" in seed_knobs.task_types:
        search_axis = "conflict_rows"
        values = [2, 3, 4, 5]
    elif seed_knobs.scatter_multi_needles and seed_knobs.num_distractors:
        search_axis = "scatter_decoys"
        values = [2, 3, 4, 5, 6, 8]
    elif seed_knobs.scatter_multi_needles:
        search_axis = "scatter_only"
        values = [1]
    elif seed_knobs.num_distractors is not None:
        search_axis = "decoys"
        values = [2, 3, 4, 5, 6, 7, 8]
    else:
        search_axis = "decoys"
        values = [2, 3, 4, 5, 6, 8]

    best_knobs = seed_knobs
    best_row = seed_row
    best_score = seed_row.get("score") if seed_row else None

    for v in values:
        if search_axis == "conflict_rows":
            knobs = KnobSettings(
                f"conflict_rows_{v}",
                task_types=["addr_val_conflict"],
                synthetic_conflict_rows=v,
                scatter_multi_needles=False,
            )
        elif search_axis == "scatter_decoys":
            knobs = KnobSettings(
                f"scatter_decoys_{v}",
                scatter_multi_needles=True,
                num_distractors=v,
            )
        elif search_axis == "scatter_only":
            knobs = KnobSettings("scatter", scatter_multi_needles=True)
        else:
            knobs = KnobSettings(f"decoys_{v}", num_distractors=v)

        run_id = f"P2_{knobs.label}"
        run_dir = output_root / "phase2" / run_id
        cfg = build_arena_config(knobs, train_t=train_t, steps=steps)
        exit_code = run_arena_pair(cfg=cfg, run_dir=run_dir, variants=["dense_flash", "linear"])
        row = record_run(
            phase="2",
            run_id=run_id,
            knobs=knobs,
            run_dir=run_dir,
            steps=steps,
            exit_code=exit_code,
            csv_path=csv_path,
        )
        if row.get("score") is not None and (best_score is None or row["score"] > best_score):
            best_score = row["score"]
            best_knobs = knobs
            best_row = row

    print(f"  Phase 2 winner: {best_knobs.label}  score={best_score}")
    return best_knobs, best_row


def run_phase3(
    *,
    output_root: Path,
    train_t: int,
    steps: int,
    csv_path: Path,
    final_knobs: KnobSettings,
) -> dict:
    print("\n=== Phase 3: confirm finalists (official holdout) ===")
    run_dir = output_root / "phase3_confirm"
    cfg = build_arena_config(final_knobs, train_t=train_t, steps=steps, validate_every=max(500, steps // 10))
    exit_code = run_arena_pair(
        cfg=cfg,
        run_dir=run_dir,
        variants=["dense_flash", "linear", "local_window64"],
    )
    row = record_run(
        phase="3",
        run_id="confirm",
        knobs=final_knobs,
        run_dir=run_dir,
        steps=steps,
        exit_code=exit_code,
        csv_path=csv_path,
        mid=False,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Dense vs linear gap calibration (3 phases)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-t", type=int, default=512)
    parser.add_argument("--screen-steps", type=int, default=SCREEN_STEPS)
    parser.add_argument("--confirm-steps", type=int, default=CONFIRM_STEPS)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 0], default=0, help="0 = all phases")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "results.csv"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    winner_knobs: KnobSettings | None = None
    winner_row: dict | None = None
    final_knobs: KnobSettings | None = None
    phase3_row: dict | None = None

    run_all = args.phase == 0
    if run_all or args.phase == 1:
        if not args.skip_phase1:
            winner_knobs, winner_row = run_phase1(
                output_root=args.output_root,
                train_t=args.train_t,
                steps=args.screen_steps,
                csv_path=csv_path,
            )

    if run_all or args.phase == 2:
        if winner_knobs is None:
            state_path = args.output_root / "phase1_winner.json"
            if state_path.exists():
                wd = json.loads(state_path.read_text(encoding="utf-8"))
                kd = wd.get("knobs", {})
                winner_knobs = KnobSettings(
                    label=kd.get("label", "baseline"),
                    scatter_multi_needles=kd.get("scatter_multi_needles"),
                    num_distractors=kd.get("num_distractors"),
                    synthetic_decoy_addrs=kd.get("synthetic_decoy_addrs"),
                    needle_depths=kd.get("needle_depths"),
                    task_types=kd.get("task_types"),
                    synthetic_conflict_rows=kd.get("synthetic_conflict_rows"),
                )
                winner_row = wd.get("row")
        if winner_knobs and not args.skip_phase2:
            final_knobs, winner_row = run_phase2(
                output_root=args.output_root,
                train_t=args.train_t,
                steps=args.screen_steps,
                csv_path=csv_path,
                seed_knobs=winner_knobs,
                seed_row=winner_row,
            )

    if final_knobs is None:
        final_knobs = winner_knobs or KnobSettings("baseline")

    if winner_knobs:
        (args.output_root / "phase1_winner.json").write_text(
            json.dumps({"knobs": winner_knobs.to_dict(), "row": winner_row}, indent=2),
            encoding="utf-8",
        )

    if run_all or args.phase == 3:
        if not args.skip_phase3:
            phase3_row = run_phase3(
                output_root=args.output_root,
                train_t=args.train_t,
                steps=args.confirm_steps,
                csv_path=csv_path,
                final_knobs=final_knobs,
            )

    summary = {
        "kind": "gap_calibration",
        "timestamp": stamp,
        "train_t": args.train_t,
        "screen_steps": args.screen_steps,
        "confirm_steps": args.confirm_steps,
        "phase1_winner": winner_knobs.to_dict() if winner_knobs else None,
        "final_knobs": final_knobs.to_dict() if final_knobs else None,
        "phase3_official": phase3_row,
        "results_csv": str(csv_path),
    }
    summary_path = args.output_root / f"gap_calibration_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Gap calibration complete ===")
    print(f"  results: {csv_path}")
    print(f"  summary: {summary_path}")
    if phase3_row:
        d = phase3_row.get("dense_acc")
        l = phase3_row.get("linear_acc")
        w = phase3_row.get("local_acc")
        print(
            f"  Phase 3 official: dense={d} linear={l} local={w} gap={phase3_row.get('gap')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
