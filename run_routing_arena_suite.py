#!/usr/bin/env python3

"""

Routing attention arena @ T=2048 — pointer_unique classic NIAH.



Fair head-to-head: dense_flash baseline + key_vector + learned_address.



Training protocol

-----------------

Prerequisite: dense_flash L2 gate checkpoint @ T=2048 (6L, ~100% pointer_unique).



dense_flash

  Official eval on the full standardized holdout (300 samples) using the dense checkpoint.



key_vector_k32

  Stage B only — Load dense trunk Q/K/V; swap to KeyVectorSparseAttention (top-k=32).

  Retrieval = head-mean Q/K dot-product (no RouterMLP, no address projections).

  Full trunk sparse LM fine-tune adapts meat weights to the sparse policy.



learned_address_k32

  Stage A.5b — Address index pretrain (InfoNCE on dense teacher caches).

  Stage B    — Load dense trunk + address index; joint sparse top-k LM fine-tune.



Eval methodology

----------------

- Official results: full 300-sample holdout (never used in training; holdout_seed ≠ train seed).

- Mid-train validation: stratified subsample at validate_every (faster feedback).



Usage

-----

  python run_routing_arena_suite.py --dry-run

  python run_routing_arena_suite.py

"""



from __future__ import annotations



import argparse

import json

import logging

import sys

import traceback

from datetime import datetime, timezone

from pathlib import Path



import torch



ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))



from routing_attention.benchmarks.long_context.production_backends import (

    assert_production_backends_available,

)

from routing_attention.benchmarks.long_context.routing_arena import (
    ARENA_VARIANTS,
    build_arena_experiment_config,

    init_arena_runtime,

    load_routing_arena_config,

    resolve_dense_checkpoint,

    run_address_index_pretrain,

    run_dense_flash_baseline,

    run_dense_flash_finetune,

    run_key_vector_k32,

    run_learned_address_k32,

    run_attention_baseline,

    BASELINE_ATTENTION_VARIANTS,

)

from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram

from routing_attention.models.fast_attention import backend_status



logger = logging.getLogger("routing_arena_suite")



ADDRESS_INDEX_VARIANTS = ("learned_address_k32",)





def preflight(dry_run: bool) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = collect_device_info(device)

    info.update(backend_status())

    info["dry_run"] = dry_run

    print("=== Routing arena preflight ===")

    for k, v in info.items():

        print(f"  {k}: {v}")

    if info["device_type"] != "cuda":

        print("WARNING: CUDA not available.")

    if not info.get("fla_linear"):

        print("ERROR: flash-linear-attention required.")

        sys.exit(1)

    try:

        assert_production_backends_available()

    except RuntimeError as exc:

        print(f"ERROR: {exc}")

        sys.exit(1)

    print()

    return info





def main() -> None:

    parser = argparse.ArgumentParser(description="Routing attention arena @ T=2048")

    parser.add_argument("--dry-run", action="store_true", help="Short smoke run (~40 sparse steps)")

    parser.add_argument(

        "--config",

        type=Path,

        default=ROOT / "configs" / "routing_arena_t2048.yaml",

    )

    parser.add_argument(

        "--variants",

        nargs="+",

        default=list(ARENA_VARIANTS),

        choices=list(ARENA_VARIANTS),

    )

    parser.add_argument("--n-layers", type=int, default=None, help="Override trunk depth (6 or 4)")

    parser.add_argument("--dense-checkpoint", type=Path, default=None)

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=ROOT / "experiments" / "Experiment_7" / "routing_arena_t2048",

    )

    parser.add_argument(

        "--force-index-pretrain",

        action="store_true",

        help="Re-collect attention caches and retrain address index",

    )

    parser.add_argument("--skip-index-pretrain", action="store_true")

    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO, format="%(message)s")



    arena_cfg = load_routing_arena_config(args.config)

    train_t = int(arena_cfg["train_context_length"])

    n_layers = args.n_layers or int(arena_cfg.get("n_layers", 6))

    holdout_cfg = arena_cfg.get("holdout", {})

    dense_gate_min = float(arena_cfg.get("dense_gate_min", 0.0))
    if args.dry_run:
        dense_gate_min = 0.0

    variants_to_run = list(args.variants)
    if dense_gate_min > 0 and "dense_flash" in variants_to_run:
        variants_to_run = ["dense_flash"] + [v for v in variants_to_run if v != "dense_flash"]

    preflight_info = preflight(args.dry_run)

    config = build_arena_experiment_config(

        arena_cfg,

        dry_run=args.dry_run,

        n_layers=n_layers,

    )



    out_dir = args.output_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    index_dir = out_dir / "index_checkpoints"



    if arena_cfg.get("dense_train_from_scratch"):
        dense_ckpt = None
    else:
        dense_ckpt = resolve_dense_checkpoint(
            train_t,
            n_layers=n_layers,
            explicit=args.dense_checkpoint or arena_cfg.get("dense_checkpoint"),
        )



    print("=== Routing arena plan ===")

    print(f"  T={train_t}  n_layers={n_layers}")

    print(f"  variants={args.variants}")

    print(f"  dense_checkpoint={dense_ckpt if dense_ckpt is not None else 'None (from scratch)'}")

    print(f"  dry_run={args.dry_run}")

    print(f"  holdout_official_target={holdout_cfg.get('total_samples', 'per-cell default')}")

    if dense_gate_min > 0 and not args.dry_run:
        print(f"  dense_gate_min={dense_gate_min * 100:.0f}%")

    if args.dry_run:

        dry = arena_cfg.get("dry_run", {})

        print(f"  dry sparse steps={dry.get('sparse_finetune_steps', 40)}")

        print(f"  dry address index={dry.get('address_index_steps', 50)}")

        print(f"  dry mid_train_per_cell={dry.get('mid_train_samples_per_cell', 4)}")

    print()



    device = init_arena_runtime(config)

    log = logging.getLogger("routing_arena.train")



    results: dict[str, dict] = {}

    errors: list[str] = []

    skip_rest_after_dense_gate = False

    address_idx = None

    if not args.skip_index_pretrain and any(v in ADDRESS_INDEX_VARIANTS for v in args.variants):

        try:

            address_idx = run_address_index_pretrain(

                config,

                dense_ckpt,

                train_t,

                index_dir,

                device,

                dry_run=args.dry_run,

                force_refresh=args.force_index_pretrain,

            )

        except Exception:

            errors.append(f"address_index_pretrain: {traceback.format_exc()}")

            print(traceback.format_exc())



    for var in variants_to_run:

        print(f"\n########## Variant: {var} ##########")

        if skip_rest_after_dense_gate:
            print(
                f"SKIP {var}: dense_flash official gate below "
                f"{dense_gate_min * 100:.0f}% threshold"
            )
            results[var] = {
                "variant": var,
                "status": "skipped_dense_gate",
                "dense_gate_min": dense_gate_min,
            }
            continue

        reset_peak_vram(device)

        try:

            if var == "dense_flash":

                if arena_cfg.get("dense_finetune_on_task"):

                    payload = run_dense_flash_finetune(

                        config,

                        train_t=train_t,

                        dense_ckpt=dense_ckpt,

                        device=device,

                        log=log,

                    )

                else:

                    payload = run_dense_flash_baseline(

                        config,

                        train_t=train_t,

                        dense_ckpt=dense_ckpt,

                        device=device,

                    )

            elif var == "key_vector_k32":

                payload = run_key_vector_k32(

                    config,

                    train_t=train_t,

                    dense_ckpt=dense_ckpt,

                    device=device,

                    log=log,

                )

            elif var == "learned_address_k32":

                if address_idx is None or not Path(address_idx).exists():

                    raise FileNotFoundError(

                        "learned_address_k32 requires Stage A.5b address index checkpoint"

                    )

                payload = run_learned_address_k32(

                    config,

                    train_t=train_t,

                    dense_ckpt=dense_ckpt,

                    address_idx=Path(address_idx),

                    device=device,

                    log=log,

                )

            elif var in BASELINE_ATTENTION_VARIANTS:

                payload = run_attention_baseline(

                    config,

                    var,

                    train_t=train_t,

                    dense_ckpt=dense_ckpt,

                    device=device,

                    log=log,

                )

            else:

                raise ValueError(f"Unknown variant: {var}")



            ev = payload.get("eval_official") or payload.get("eval", {})

            gate = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))

            print(

                f"OK {var}: official_gate={float(gate or 0) * 100:.2f}% "

                f"({ev.get('primary_gate_correct', ev.get('correct'))}/"

                f"{ev.get('primary_gate_total', ev.get('total'))}) "

                f"[holdout={ev.get('holdout_samples', '?')}]"

            )

            results[var] = payload

            if (
                var == "dense_flash"
                and dense_gate_min > 0
                and arena_cfg.get("dense_finetune_on_task")
            ):
                gate_acc = float(gate or 0)
                if gate_acc < dense_gate_min:
                    skip_rest_after_dense_gate = True
                    print(
                        f"\nDENSE GATE FAILED: official_gate={gate_acc * 100:.2f}% "
                        f"< {dense_gate_min * 100:.0f}% — skipping remaining variants"
                    )
        except Exception:

            err = traceback.format_exc()

            errors.append(f"{var}: {err}")

            results[var] = {"variant": var, "status": "error", "traceback": err}

            print(err)



        if device.type == "cuda":

            torch.cuda.empty_cache()



    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    tag = "dry_run" if args.dry_run else "full"

    summary = {

        "timestamp": datetime.now(timezone.utc).isoformat(),

        "kind": "routing_arena_suite",

        "dry_run": args.dry_run,

        "train_context_length": train_t,

        "n_layers": n_layers,

        "dense_checkpoint": str(dense_ckpt),

        "address_index_checkpoint": str(address_idx) if address_idx else None,

        "holdout": config.get("holdout", holdout_cfg),

        "variants": args.variants,

        "preflight": preflight_info,

        "results": results,

        "errors": errors,

    }

    summary_path = out_dir / f"routing_arena_{tag}_{stamp}.json"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_path = out_dir / "latest.json"

    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



    print(f"\n=== Routing arena summary ===")

    print(f"  wrote: {summary_path}")

    for var, payload in results.items():

        if payload.get("status") == "error":

            print(f"  {var}: ERROR")

        elif payload.get("status") == "skipped_dense_gate":

            print(f"  {var}: SKIPPED (dense gate)")

        else:

            ev = payload.get("eval_official") or payload.get("eval", {})

            acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0))

            print(

                f"  {var}: official_gate={float(acc) * 100:.2f}% "

                f"({ev.get('primary_gate_correct', ev.get('correct'))}/"

                f"{ev.get('primary_gate_total', ev.get('total'))})"

            )



    if errors:

        print(f"  errors={len(errors)}")

        sys.exit(1)





if __name__ == "__main__":

    main()

