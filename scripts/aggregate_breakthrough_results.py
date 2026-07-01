#!/usr/bin/env python3
"""Aggregate breakthrough experiment outputs into a unified results table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from learned_address_proof_common import (
    BREAKTHROUGH_CLAIM,
    BREAKTHROUGH_OUTPUT,
    official_accuracy,
    post_phase_c_recall,
    recall_at_k,
)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _systems_rows(base: Path) -> list[dict]:
    rows: list[dict] = []
    search_roots = [base, base.parent / "learned_address_proof_cell"]
    for root in search_roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("systems_benchmark.json")):
            data = _load_json(p)
            if isinstance(data, dict):
                for row in data.get("rows", []):
                    row = dict(row)
                    row["source"] = str(p.parent)
                    rows.append(row)
    return rows


def _variant_row(
    *,
    experiment: str,
    variant: str,
    train_t: int | None,
    seed: int | None,
    payload: dict,
    recall_b: float | None = None,
    recall_c: float | None = None,
) -> dict[str, Any]:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    return {
        "experiment": experiment,
        "variant": variant,
        "train_t": train_t,
        "seed": seed,
        "accuracy": official_accuracy(payload),
        "recall_at_k_after_b": recall_b,
        "recall_at_k_after_c": recall_c,
        "by_needle_depth": ev.get("by_needle_depth"),
        "peak_vram_mb": payload.get("peak_vram_mb"),
    }


def build_table(input_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    proof_root = input_dir.parent / "learned_address_proof_cell"
    proof_manifest = _load_json(proof_root / "proof_cell_full" / "manifest.json")
    if isinstance(proof_manifest, dict):
        recall_k = int(proof_manifest.get("recall_k", 128))
        b_recall = proof_manifest.get("phase_b", {}).get(f"recall@{recall_k}")
        for variant, payload in (proof_manifest.get("phase_c") or {}).items():
            if isinstance(payload, dict):
                rows.append(
                    _variant_row(
                        experiment="proof_cell",
                        variant=variant,
                        train_t=int(proof_manifest.get("train_t", 2048)),
                        seed=45,
                        payload=payload,
                        recall_b=b_recall if variant == "learned_address_k32" else None,
                        recall_c=post_phase_c_recall(payload, recall_k),
                    )
                )
        rows.append(
            {
                "experiment": "proof_cell",
                "variant": "dense_flash",
                "train_t": 2048,
                "seed": 45,
                "accuracy": proof_manifest.get("dense_accuracy"),
                "recall_at_k_after_b": b_recall,
                "recall_at_k_after_c": None,
                "by_needle_depth": depth_from_phase_a(proof_manifest.get("phase_a")),
            }
        )

    tag = "dry" if dry_run else "full"
    linear = _load_json(input_dir / f"phase1_linear_{tag}_latest.json")
    if isinstance(linear, dict):
        rows.append(
            {
                "experiment": "phase1_linear",
                "variant": "linear",
                "train_t": 2048,
                "seed": 45,
                "accuracy": linear.get("linear_accuracy"),
                "by_needle_depth": linear.get("by_needle_depth"),
            }
        )

    sweep = _load_json(proof_root / f"sweep_{tag}_latest.json") or _load_json(
        proof_root / f"sweep_{tag}" / "sweep_manifest.json"
    )
    if isinstance(sweep, dict):
        for row in sweep.get("rows", []):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "experiment": "phase1_sweep",
                    "variant": "learned_address_k32",
                    "train_t": 2048,
                    "seed": 45,
                    "b_steps": row.get("b_steps"),
                    "accuracy": row.get("learned_address_accuracy"),
                    "recall_at_k_after_b": row.get("recall"),
                }
            )

    curriculum = _load_json(proof_root / f"curriculum_{tag}_latest.json")
    if isinstance(curriculum, dict):
        for t, cell in (curriculum.get("cells") or {}).items():
            if not isinstance(cell, dict):
                continue
            rows.append(
                {
                    "experiment": "phase1_curriculum",
                    "variant": "learned_address_k32",
                    "train_t": int(t),
                    "seed": 45,
                    "accuracy": cell.get("learned_address_accuracy"),
                    "dense_accuracy": cell.get("dense_accuracy"),
                    "recall_at_k_after_b": (cell.get("phase_b") or {}).get("recall@128"),
                }
            )

    for cell_path in sorted((input_dir / "phase2_seed_repro").rglob("cell.json")):
        cell = _load_json(cell_path)
        if not isinstance(cell, dict):
            continue
        seed = cell.get("seed")
        for variant, metrics in (cell.get("metrics") or {}).items():
            rows.append(
                {
                    "experiment": "phase2_seed_repro",
                    "variant": variant,
                    "train_t": cell.get("train_t"),
                    "seed": seed,
                    **metrics,
                    "recall_at_k_after_b": cell.get("recall@128_after_b") if variant == "learned_address_k32" else None,
                }
            )

    hard = _load_json(input_dir / "phase2_hard_cell" / "hard_cell_d1" / tag / "cell.json")
    if isinstance(hard, dict):
        for variant, metrics in (hard.get("metrics") or {}).items():
            rows.append({"experiment": "phase2_hard_cell_d1", "variant": variant, "seed": 45, **metrics})

    systems = _systems_rows(input_dir)
    latency_rows = [
        {
            "experiment": "systems",
            "variant": r.get("variant"),
            "context_length": r.get("context_length"),
            "latency_ms": r.get("latency_ms"),
            "peak_vram_mb": r.get("peak_vram_mb"),
            "tokens_per_sec": r.get("tokens_per_sec"),
            "error": r.get("error"),
        }
        for r in systems
    ]

    return {
        "dry_run": dry_run,
        "claim": BREAKTHROUGH_CLAIM,
        "quality_rows": rows,
        "systems_rows": latency_rows,
        "summary": summarize(rows, latency_rows),
    }


def depth_from_phase_a(phase_a: dict | None) -> dict | None:
    if not phase_a:
        return None
    ev = phase_a.get("eval_official") or phase_a.get("eval") or {}
    return ev.get("by_needle_depth")


def summarize(quality: list[dict], systems: list[dict]) -> dict[str, Any]:
    proof = [r for r in quality if r.get("experiment") == "proof_cell" and r.get("train_t") == 2048]
    la = next((r for r in proof if r.get("variant") == "learned_address_k32"), None)
    dense = next((r for r in proof if r.get("variant") == "dense_flash"), None)
    kv = next((r for r in proof if r.get("variant") == "key_vector_k32"), None)
    local = next((r for r in proof if r.get("variant") == "local_window64"), None)
    lat_16k = [
        r for r in systems if r.get("context_length") == 16384 and not r.get("error")
    ]
    dense_lat = next((r for r in lat_16k if r.get("variant") == "dense_flash"), None)
    la_lat = next((r for r in lat_16k if r.get("variant") == "learned_address_k32"), None)
    speed_ratio = None
    if dense_lat and la_lat and dense_lat.get("latency_ms") and la_lat.get("latency_ms"):
        speed_ratio = float(dense_lat["latency_ms"]) / float(la_lat["latency_ms"])

    return {
        "proof_cell_learned_address_acc": la.get("accuracy") if la else None,
        "proof_cell_dense_acc": dense.get("accuracy") if dense else None,
        "proof_cell_key_vector_acc": kv.get("accuracy") if kv else None,
        "proof_cell_local64_acc": local.get("accuracy") if local else None,
        "learned_minus_key_vector_pp": (
            (float(la["accuracy"]) - float(kv["accuracy"])) * 100.0
            if la and kv and la.get("accuracy") is not None and kv.get("accuracy") is not None
            else None
        ),
        "speedup_vs_dense_at_16k": speed_ratio,
        "breakthrough_quality_parity": (
            la.get("accuracy") is not None
            and dense.get("accuracy") is not None
            and abs(float(la["accuracy"]) - float(dense["accuracy"])) <= 0.03
        )
        if la and dense
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BREAKTHROUGH_OUTPUT)
    parser.add_argument("--output", type=Path, default=BREAKTHROUGH_OUTPUT / "results_table.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    table = build_table(args.input, dry_run=args.dry_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(table.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
