#!/usr/bin/env python3
"""
Learned-address proof cell suite — validate the 3-phase routing protocol.

Phases (recommended order):
  proof_cell  — T=2048 canonical cell (A dense → B address index → C sparse variants + systems)
  sweep       — Phase B step sweep (2k/5k/10k/20k) with fixed C=20k K=128
  curriculum  — T=2048→4096→8192 learned-address protocol at each length

Every phase supports --dry-run first (short smoke steps) before full training.

Usage:
  python run_learned_address_proof_cell_suite.py --phase proof_cell --dry-run
  python run_learned_address_proof_cell_suite.py --phase proof_cell
  python run_learned_address_proof_cell_suite.py --phase all --dry-run
  python run_learned_address_proof_cell_suite.py --phase all
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from learned_address_proof_common import (
    CONFIG_PATH,
    CURRICULUM_LENGTHS,
    DEFAULT_DENSE_CKPT,
    OUTPUT_ROOT,
    PHASE_C_VARIANTS,
    PROOF_VARIANTS,
    SWEEP_B_STEPS,
    SYSTEMS_LENGTHS,
    SYSTEMS_VARIANTS,
    measure_address_recall,
    measure_key_vector_recall,
    official_accuracy,
    recall_at_k,
    write_json,
)
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.index_pretrain import (
    address_index_checkpoint_path,
    pretrain_addresses_on_dense_checkpoint,
)
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_attention_baseline,
    run_dense_flash_eval_from_checkpoint,
    run_dense_flash_finetune,
    run_key_vector_k32,
    run_learned_address_k32,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.checkpoint import load_checkpoint
from routing_attention.utils.cuda import configure_cuda_training

VERIFY_SCRIPT = ROOT / "scripts" / "verify_learned_address_proof_cell.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def preflight(dry_run: bool) -> dict:
    configure_cuda_training({"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== learned_address_proof_cell preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — training will be very slow.")
    assert_production_backends_available(list(PROOF_VARIANTS))
    print()
    return info


def run_verify() -> None:
    if not VERIFY_SCRIPT.exists():
        return
    print("=== Config verification ===")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print()


def _load_arena(
    dry_run: bool,
    *,
    train_t: int | None = None,
    b_steps: int | None = None,
    config_path: Path | None = None,
) -> tuple[dict, dict, dict]:
    arena_cfg = load_routing_arena_config(config_path or CONFIG_PATH)
    if train_t is not None:
        arena_cfg = copy.deepcopy(arena_cfg)
        arena_cfg["train_context_length"] = train_t
        bench = dict(arena_cfg.get("long_context_benchmark", {}))
        bench["context_lengths"] = [train_t]
        arena_cfg["long_context_benchmark"] = bench
    if b_steps is not None:
        arena_cfg = copy.deepcopy(arena_cfg)
        arena_cfg.setdefault("index_pretrain", {})["address_index_steps"] = b_steps
        arena_cfg.setdefault("learned_address", {})["address_index_steps"] = b_steps
    n_layers = int(arena_cfg.get("n_layers", 4))
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    return arena_cfg, config, arena_cfg.get("proof_cell", {})


def _write_json(path: Path, payload: dict) -> None:
    write_json(path, payload)


def _upstream_gates_block(arena_cfg: dict) -> bool:
    return bool(arena_cfg.get("proof_cell", {}).get("block_on_upstream_gates", False))


def _phase_b_recall_from_meta(meta: dict, recall_k: int) -> float | None:
    holdout = meta.get("holdout_recall") or {}
    return recall_at_k(holdout, recall_k)


def run_phase_b(
    *,
    config: dict,
    arena_cfg: dict,
    dense_ckpt: Path,
    train_t: int,
    device: torch.device,
    out_dir: Path,
    dry_run: bool,
    force_refresh: bool,
) -> dict[str, Any]:
    proof = arena_cfg.get("proof_cell", {})
    recall_k = int(proof.get("recall_k", config.get("router", {}).get("top_k", 128)))
    index_dir = out_dir / "index_checkpoints"
    addr_path = address_index_checkpoint_path(index_dir, train_t)

    print(f"\n########## Phase B: address index pretrain T={train_t} ##########")
    meta = pretrain_addresses_on_dense_checkpoint(
        config,
        dense_ckpt,
        train_t,
        addr_path,
        device,
        dry_run=dry_run,
        force_refresh_cache=force_refresh,
    )
    recall = _phase_b_recall_from_meta(meta, recall_k)
    recall_min = float(proof.get("phase_b_recall_min", 0.0))
    block_on_recall = bool(proof.get("block_on_recall_gate", False))
    gate_passed = not block_on_recall or recall is None or recall >= recall_min

    if recall is not None:
        print(f"  Phase B Recall@{recall_k}={recall * 100:.2f}%  gate>={recall_min * 100:.0f}%  passed={gate_passed}")
    else:
        print("  Phase B Recall@K: n/a")

    # Key-vector recall baseline on same holdout cache (dense Q/K geometry).
    kv_recall_metrics: dict[str, Any] = {}
    holdout_cache = meta.get("holdout_cache")
    if holdout_cache and Path(holdout_cache).exists():
        try:
            from experiments.experiment_7 import _build_variant_model

            dense_model, _ = _build_variant_model(
                config, "dense_flash", device, train_t, dense_checkpoint=dense_ckpt
            )
            dense_model.eval()
            kv_recall_metrics = measure_key_vector_recall(
                dense_model,
                holdout_cache,
                device=device,
                recall_k=recall_k,
                n_layers=int(config["model"]["n_layers"]),
                dry_run=dry_run,
            )
            kv_recall = recall_at_k(kv_recall_metrics, recall_k)
            print(f"  Key-vector Recall@{recall_k} (dense Q/K)={kv_recall * 100:.2f}%" if kv_recall else "  Key-vector Recall: n/a")
            del dense_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            kv_recall_metrics = {"error": str(exc)}
            print(f"  Key-vector recall baseline failed: {exc}")

    payload = {
        "phase": "B",
        "train_t": train_t,
        "address_index_checkpoint": str(addr_path),
        "meta": meta,
        f"recall@{recall_k}": recall,
        "key_vector_recall": kv_recall_metrics,
        "gate": {
            "recall_min": recall_min,
            "passed": gate_passed,
        },
    }
    _write_json(out_dir / "phase_b.json", payload)
    return payload


def run_phase_c_variant(
    variant: str,
    *,
    config: dict,
    train_t: int,
    dense_ckpt: Path,
    address_idx: Path | None,
    device: torch.device,
    log: logging.Logger,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    print(f"\n########## Phase C: {variant} T={train_t} ##########")
    reset_peak_vram(device)
    variant_dir = out_dir / "phase_c" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    if variant == "key_vector_k32":
        payload = run_key_vector_k32(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            device=device,
            log=log,
            top_k=int(config.get("key_vector", {}).get("top_k") or config.get("router", {}).get("top_k", 128)),
        )
    elif variant == "learned_address_k32":
        if address_idx is None or not address_idx.exists():
            raise FileNotFoundError(f"learned_address requires address index: {address_idx}")
        payload = run_learned_address_k32(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            address_idx=address_idx,
            device=device,
            log=log,
        )
    elif variant in ("local_window64", "local_window256", "linear"):
        payload = run_attention_baseline(
            config,
            variant,
            train_t=train_t,
            dense_ckpt=None if variant == "linear" else None,
            device=device,
            log=log,
        )
    else:
        raise ValueError(f"Unsupported Phase C variant: {variant}")

    acc = official_accuracy(payload)
    vram = peak_vram_mb(device)
    payload["peak_vram_mb"] = vram
    if acc is not None:
        print(f"  official_acc={acc * 100:.2f}%  peak_vram={vram}MB")

    # Recall after Phase C for learned-address (index may have drifted during joint finetune).
    if variant == "learned_address_k32" and address_idx is not None:
        holdout_cache = None
        phase_b_path = out_dir / "phase_b.json"
        if phase_b_path.exists():
            holdout_cache = json.loads(phase_b_path.read_text(encoding="utf-8")).get("meta", {}).get("holdout_cache")
        if holdout_cache and Path(holdout_cache).exists():
            try:
                from experiments.common import build_address_book
                from routing_attention.models.learned_address import attach_address_book_to_model, ensure_address_book_on_model

                book = build_address_book(config).to(device)
                load_checkpoint(address_idx, book, device=device, strict=False)
                recall_k = int(config.get("router", {}).get("top_k", 128))
                post_c_recall = measure_address_recall(
                    book,
                    holdout_cache,
                    device=device,
                    recall_k=recall_k,
                    n_layers=int(config["model"]["n_layers"]),
                    dry_run=dry_run,
                )
                payload["post_phase_c_recall"] = post_c_recall
                r = recall_at_k(post_c_recall, recall_k)
                if r is not None:
                    print(f"  post-C address Recall@{recall_k}={r * 100:.2f}%")
                del book
            except Exception as exc:
                payload["post_phase_c_recall"] = {"error": str(exc)}

    _write_json(variant_dir / "latest.json", payload)
    return payload


def run_systems_benchmark(
    *,
    config: dict,
    train_t: int,
    dense_ckpt: Path,
    address_idx: Path | None,
    device: torch.device,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    print(f"\n########## Systems latency + VRAM ##########")
    from experiments.experiment_7 import _build_variant_model
    from routing_attention.benchmarks.long_context.routing_arena import _load_address_index_into_model

    bench_cfg = _resolve_synthetic_bench_cfg(config, train_t)
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=[])
    rows: list[dict] = []

    for ctx_t in SYSTEMS_LENGTHS:
        print(f"\n  context T={ctx_t}")
        for variant in SYSTEMS_VARIANTS:
            reset_peak_vram(device)
            try:
                model, var_config = _build_variant_model(
                    config,
                    variant,
                    device,
                    ctx_t,
                    dense_checkpoint=dense_ckpt,
                )
                if variant == "learned_address_k32":
                    if address_idx is None or not address_idx.exists():
                        raise FileNotFoundError("missing address index for learned_address systems bench")
                    _load_address_index_into_model(model, var_config, address_idx, device)
                model.eval()
                lat = evaluator.benchmark_forward_latency(
                    model,
                    device=device,
                    context_length=ctx_t,
                    warmup=1 if dry_run else 2,
                    runs=2 if dry_run else 5,
                )
                vram = peak_vram_mb(device)
                row = {
                    "variant": variant,
                    "context_length": ctx_t,
                    "latency_ms": lat.get("latency_ms"),
                    "tokens_per_sec": lat.get("tokens_per_sec"),
                    "peak_vram_mb": vram,
                    "error": lat.get("error"),
                }
                print(f"    {variant}: {lat.get('latency_ms')} ms  vram={vram}MB")
            except Exception as exc:
                row = {"variant": variant, "context_length": ctx_t, "error": str(exc)[:400]}
                print(f"    {variant}: ERROR {exc}")
            rows.append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {"dry_run": dry_run, "timestamp": _now(), "rows": rows}
    _write_json(out_dir / "systems_benchmark.json", summary)
    return summary


def run_proof_cell(
    *,
    dry_run: bool,
    dense_checkpoint: Path | None,
    force_index: bool,
) -> dict[str, Any]:
    tag = "dry" if dry_run else "full"
    out_dir = OUTPUT_ROOT / f"proof_cell_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_cfg, config, proof = _load_arena(dry_run)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg["n_layers"])
    recall_k = int(proof.get("recall_k", 128))
    dense_gate_min = float(arena_cfg.get("dense_gate_min", 0.95))
    gap_pp = float(proof.get("phase_c_dense_gap_pp", 3.0))

    bench = _resolve_synthetic_bench_cfg(config, train_t)
    print("=== Proof cell plan ===")
    print(f"  T={train_t}  n_layers={n_layers}  decoys={bench.synthetic_decoy_keys}")
    print(f"  dense_steps={config['transformer'].get('dense_pretrain_steps')}")
    print(f"  phase_b_steps={config.get('index_pretrain', {}).get('address_index_steps')}")
    print(f"  phase_c_steps={config.get('learned_address', {}).get('joint_finetune_steps')}")
    print(f"  K={recall_k}  dry_run={dry_run}")
    print(f"  output={out_dir}")
    print()

    device = init_arena_runtime(config)
    log = logging.getLogger("proof_cell")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dense_ckpt_path = ckpt_dir / ("dense_flash_dry.pt" if dry_run else "dense_flash.pt")

    results: dict[str, Any] = {
        "kind": "learned_address_proof_cell",
        "dry_run": dry_run,
        "timestamp": _now(),
        "train_t": train_t,
        "n_layers": n_layers,
        "recall_k": recall_k,
    }
    errors: list[str] = []

    # Phase A — dense teacher
    print("\n########## Phase A: dense teacher ##########")
    dense_ckpt = dense_checkpoint
    dense_eval_only = dense_ckpt is not None and dense_ckpt.exists()
    try:
        if dense_eval_only:
            print(f"  eval-only from {dense_ckpt}")
            phase_a = run_dense_flash_eval_from_checkpoint(
                config, train_t=train_t, dense_ckpt=dense_ckpt, device=device, log=log
            )
        else:
            phase_a = run_dense_flash_finetune(
                config,
                train_t=train_t,
                dense_ckpt=None,
                device=device,
                log=log,
                save_checkpoint_path=dense_ckpt_path,
            )
            saved = phase_a.get("saved_dense_checkpoint")
            dense_ckpt = Path(saved) if saved else dense_ckpt_path
        dense_acc = official_accuracy(phase_a)
        print(f"  dense_acc={dense_acc * 100:.2f}%" if dense_acc is not None else "  dense_acc=n/a")
        results["phase_a"] = phase_a
        results["dense_accuracy"] = dense_acc
        results["dense_checkpoint"] = str(dense_ckpt)
    except Exception:
        err = traceback.format_exc()
        errors.append(f"phase_a: {err}")
        print(err)
        results["phase_a"] = {"status": "error", "traceback": err}
        _write_json(out_dir / "manifest.json", {**results, "errors": errors, "gates": {"passed": False}})
        return results

    dense_gate = dense_acc is not None and dense_acc >= dense_gate_min
    if not dense_gate and not dry_run:
        print(f"\nDENSE GATE FAILED: {dense_acc * 100:.2f}% < {dense_gate_min * 100:.0f}%")
        results["gates"] = {"dense_passed": False, "passed": False}
        _write_json(out_dir / "manifest.json", {**results, "errors": errors})
        return results

    if dense_ckpt is None or not Path(dense_ckpt).exists():
        raise FileNotFoundError("Dense checkpoint missing after Phase A")

    # Phase B — address index
    try:
        phase_b = run_phase_b(
            config=config,
            arena_cfg=arena_cfg,
            dense_ckpt=Path(dense_ckpt),
            train_t=train_t,
            device=device,
            out_dir=out_dir,
            dry_run=dry_run,
            force_refresh=force_index,
        )
        results["phase_b"] = phase_b
        address_idx = Path(phase_b["address_index_checkpoint"])
    except Exception:
        err = traceback.format_exc()
        errors.append(f"phase_b: {err}")
        print(err)
        results["phase_b"] = {"status": "error", "traceback": err}
        _write_json(out_dir / "manifest.json", {**results, "errors": errors, "gates": {"passed": False}})
        return results

    b_recall = phase_b.get(f"recall@{recall_k}")
    b_gate = phase_b.get("gate", {}).get("passed", True)
    if not b_gate and not dry_run:
        print(f"\nPHASE B RECALL GATE FAILED: {b_recall}")

    # Phase C — sparse variants + linear baseline
    phase_c: dict[str, Any] = {}
    for variant in PHASE_C_VARIANTS:
        try:
            phase_c[variant] = run_phase_c_variant(
                variant,
                config=config,
                train_t=train_t,
                dense_ckpt=Path(dense_ckpt),
                address_idx=address_idx,
                device=device,
                log=log,
                out_dir=out_dir,
                dry_run=dry_run,
            )
        except Exception:
            err = traceback.format_exc()
            errors.append(f"phase_c/{variant}: {err}")
            print(err)
            phase_c[variant] = {"status": "error", "traceback": err}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["phase_c"] = phase_c

    la_acc = official_accuracy(phase_c.get("learned_address_k32", {}))
    kv_acc = official_accuracy(phase_c.get("key_vector_k32", {}))
    linear_acc = official_accuracy(phase_c.get("linear", {}))
    local64 = official_accuracy(phase_c.get("local_window64", {}))
    c_gate = (
        la_acc is not None
        and dense_acc is not None
        and abs(la_acc - dense_acc) * 100.0 <= gap_pp
    )

    # Systems benchmark
    try:
        results["systems"] = run_systems_benchmark(
            config=config,
            train_t=train_t,
            dense_ckpt=Path(dense_ckpt),
            address_idx=address_idx,
            device=device,
            out_dir=out_dir,
            dry_run=dry_run,
        )
    except Exception:
        err = traceback.format_exc()
        errors.append(f"systems: {err}")
        print(err)
        results["systems"] = {"status": "error", "traceback": err}

    block_recall = bool(proof.get("block_on_recall_gate", False))
    gates = {
        "dense_passed": dense_gate,
        "phase_b_recall_passed": b_gate,
        "learned_address_near_dense": c_gate if not dry_run else None,
        "learned_address_accuracy": la_acc,
        "key_vector_accuracy": kv_acc,
        "linear_accuracy": linear_acc,
        "local_window64_accuracy": local64,
        "dense_accuracy": dense_acc,
        "phase_b_recall": b_recall,
        "infrastructure_ok": not errors,
        "passed": (not errors)
        if dry_run
        else (
            not errors
            and dense_gate
            and (not block_recall or b_gate)
            and c_gate
        ),
    }
    if la_acc is not None and kv_acc is not None:
        gates["learned_address_minus_key_vector_pp"] = (la_acc - kv_acc) * 100.0
    if la_acc is not None and dense_acc is not None:
        gates["learned_address_minus_dense_pp"] = (la_acc - dense_acc) * 100.0

    results["gates"] = gates
    results["errors"] = errors
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, results)
    _write_json(OUTPUT_ROOT / f"proof_cell_{tag}_latest.json", results)

    print("\n=== Proof cell summary ===")
    print(f"  dense:           {dense_acc * 100:.2f}%" if dense_acc else "  dense: n/a")
    print(f"  Phase B Recall:  {b_recall * 100:.2f}%" if b_recall else "  Phase B Recall: n/a")
    print(f"  learned-address: {la_acc * 100:.2f}%" if la_acc else "  learned-address: n/a")
    print(f"  key-vector:      {kv_acc * 100:.2f}%" if kv_acc else "  key-vector: n/a")
    print(f"  linear:          {linear_acc * 100:.2f}%" if linear_acc else "  linear: n/a")
    print(f"  local W=64:      {local64 * 100:.2f}%" if local64 else "  local W=64: n/a")
    print(f"  gates passed:    {gates['passed']}")
    print(f"  manifest:        {manifest_path}")

    return results


def run_sweep(*, dry_run: bool, dense_checkpoint: Path | None) -> dict[str, Any]:
    tag = "dry" if dry_run else "full"
    out_root = OUTPUT_ROOT / f"sweep_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Phase B step sweep ({tag}) ===")

    arena_base = load_routing_arena_config(CONFIG_PATH)
    proof_manifest = OUTPUT_ROOT / f"proof_cell_{tag}_latest.json"
    if (
        not dry_run
        and _upstream_gates_block(arena_base)
        and proof_manifest.exists()
    ):
        proof = json.loads(proof_manifest.read_text(encoding="utf-8"))
        if not proof.get("gates", {}).get("passed"):
            print("SKIP sweep: proof cell gates did not pass (block_on_upstream_gates=true)")
            return {"skipped": True, "reason": "proof_cell_gate_failed"}

    dense_ckpt = dense_checkpoint
    if dense_ckpt is None:
        proof_full = OUTPUT_ROOT / "proof_cell_full_latest.json"
        if proof_full.exists():
            dense_ckpt = Path(json.loads(proof_full.read_text(encoding="utf-8"))["dense_checkpoint"])
        elif DEFAULT_DENSE_CKPT.exists():
            dense_ckpt = DEFAULT_DENSE_CKPT

    rows: list[dict] = []
    b_steps = [10_000] if dry_run else SWEEP_B_STEPS
    for b_steps_val in b_steps:
        cell_dir = out_root / f"B{b_steps_val}"
        arena_cfg, config, _ = _load_arena(dry_run, b_steps=b_steps_val if not dry_run else None)
        train_t = int(arena_cfg["train_context_length"])
        device = init_arena_runtime(config)
        if dense_ckpt is None or not Path(dense_ckpt).exists():
            raise FileNotFoundError("Sweep requires dense checkpoint from proof cell or --dense-checkpoint")

        try:
            phase_b = run_phase_b(
                config=config,
                arena_cfg=arena_cfg,
                dense_ckpt=Path(dense_ckpt),
                train_t=train_t,
                device=device,
                out_dir=cell_dir,
                dry_run=dry_run,
                force_refresh=True,
            )
            address_idx = Path(phase_b["address_index_checkpoint"])
            la = run_phase_c_variant(
                "learned_address_k32",
                config=config,
                train_t=train_t,
                dense_ckpt=Path(dense_ckpt),
                address_idx=address_idx,
                device=device,
                log=logging.getLogger(f"sweep.B{b_steps_val}"),
                out_dir=cell_dir,
                dry_run=dry_run,
            )
            recall_k = int(arena_cfg.get("proof_cell", {}).get("recall_k", 128))
            row = {
                "b_steps": b_steps_val,
                "recall": phase_b.get(f"recall@{recall_k}"),
                "learned_address_accuracy": official_accuracy(la),
                "phase_b": phase_b,
                "phase_c": la,
            }
            print(f"  B={b_steps_val}: recall={row['recall']}  la_acc={row['learned_address_accuracy']}")
        except Exception as exc:
            row = {"b_steps": b_steps_val, "error": str(exc), "traceback": traceback.format_exc()}
            print(f"  B={b_steps_val}: ERROR {exc}")
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"dry_run": dry_run, "timestamp": _now(), "rows": rows}
    _write_json(out_root / "sweep_manifest.json", summary)
    _write_json(OUTPUT_ROOT / f"sweep_{tag}_latest.json", summary)
    return summary


def run_curriculum(*, dry_run: bool, dense_checkpoint: Path | None) -> dict[str, Any]:
    tag = "dry" if dry_run else "full"
    out_root = OUTPUT_ROOT / f"curriculum_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    lengths = [2048] if dry_run else CURRICULUM_LENGTHS
    print(f"\n=== T curriculum {lengths} ({tag}) ===")

    arena_base = load_routing_arena_config(CONFIG_PATH)
    if not dry_run and _upstream_gates_block(arena_base):
        sweep_manifest = OUTPUT_ROOT / "sweep_full_latest.json"
        proof_manifest = OUTPUT_ROOT / "proof_cell_full_latest.json"
        ok = False
        if proof_manifest.exists():
            ok = json.loads(proof_manifest.read_text(encoding="utf-8")).get("gates", {}).get("passed", False)
        if sweep_manifest.exists() and not ok:
            rows = json.loads(sweep_manifest.read_text(encoding="utf-8")).get("rows", [])
            ok = any(
                r.get("learned_address_accuracy", 0) >= 0.95
                for r in rows
                if isinstance(r.get("learned_address_accuracy"), (int, float))
            )
        if not ok:
            print("SKIP curriculum: proof cell / sweep gates did not pass (block_on_upstream_gates=true)")
            return {"skipped": True, "reason": "upstream_gate_failed"}

    cells: dict[str, Any] = {}
    prev_dense: Path | None = dense_checkpoint

    for train_t in lengths:
        cell_dir = out_root / f"T{train_t}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        arena_cfg, config, proof = _load_arena(dry_run, train_t=train_t)
        device = init_arena_runtime(config)
        log = logging.getLogger(f"curriculum.T{train_t}")
        ckpt_dir = cell_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dense_path = ckpt_dir / ("dense_flash_dry.pt" if dry_run else "dense_flash.pt")

        print(f"\n########## Curriculum T={train_t} ##########")
        cell: dict[str, Any] = {"train_t": train_t}

        # Phase A at this T — continue from previous dense if available and same recipe allows
        try:
            if prev_dense and prev_dense.exists() and train_t > lengths[0]:
                print(f"  Phase A warm-start from {prev_dense}")
                phase_a = run_dense_flash_finetune(
                    config,
                    train_t=train_t,
                    dense_ckpt=prev_dense,
                    device=device,
                    log=log,
                    save_checkpoint_path=dense_path,
                )
            else:
                phase_a = run_dense_flash_finetune(
                    config,
                    train_t=train_t,
                    dense_ckpt=None,
                    device=device,
                    log=log,
                    save_checkpoint_path=dense_path,
                )
            dense_ckpt = Path(phase_a.get("saved_dense_checkpoint") or dense_path)
            cell["phase_a"] = phase_a
            cell["dense_accuracy"] = official_accuracy(phase_a)
        except Exception:
            err = traceback.format_exc()
            cell["phase_a"] = {"error": err}
            cells[str(train_t)] = cell
            print(err)
            continue

        try:
            phase_b = run_phase_b(
                config=config,
                arena_cfg=arena_cfg,
                dense_ckpt=dense_ckpt,
                train_t=train_t,
                device=device,
                out_dir=cell_dir,
                dry_run=dry_run,
                force_refresh=True,
            )
            address_idx = Path(phase_b["address_index_checkpoint"])
            cell["phase_b"] = phase_b
            la = run_phase_c_variant(
                "learned_address_k32",
                config=config,
                train_t=train_t,
                dense_ckpt=dense_ckpt,
                address_idx=address_idx,
                device=device,
                log=log,
                out_dir=cell_dir,
                dry_run=dry_run,
            )
            cell["learned_address"] = la
            cell["learned_address_accuracy"] = official_accuracy(la)
        except Exception:
            err = traceback.format_exc()
            cell["error"] = err
            print(err)

        prev_dense = dense_ckpt
        cells[str(train_t)] = cell
        _write_json(cell_dir / "cell.json", cell)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"dry_run": dry_run, "timestamp": _now(), "cells": cells}
    _write_json(out_root / "curriculum_manifest.json", summary)
    _write_json(OUTPUT_ROOT / f"curriculum_{tag}_latest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned-address proof cell suite")
    parser.add_argument(
        "--phase",
        choices=("proof_cell", "sweep", "curriculum", "all"),
        default="proof_cell",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    parser.add_argument("--force-index-pretrain", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    suite_log = OUTPUT_ROOT / ("suite_dry.log" if args.dry_run else "suite_full.log")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(suite_log, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    if not args.skip_verify:
        run_verify()
    preflight(args.dry_run)

    exit_code = 0
    phases = ["proof_cell", "sweep", "curriculum"] if args.phase == "all" else [args.phase]

    for phase in phases:
        print(f"\n{'=' * 60}\nPHASE: {phase}  dry_run={args.dry_run}\n{'=' * 60}")
        try:
            if phase == "proof_cell":
                result = run_proof_cell(
                    dry_run=args.dry_run,
                    dense_checkpoint=args.dense_checkpoint,
                    force_index=args.force_index_pretrain,
                )
            elif phase == "sweep":
                result = run_sweep(dry_run=args.dry_run, dense_checkpoint=args.dense_checkpoint)
            else:
                result = run_curriculum(dry_run=args.dry_run, dense_checkpoint=args.dense_checkpoint)
            if result.get("errors") or result.get("skipped"):
                if result.get("errors"):
                    exit_code = 1
        except Exception:
            print(traceback.format_exc())
            exit_code = 1

    logging.getLogger().removeHandler(fh)
    fh.close()
    print(f"\nSuite log: {suite_log}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
