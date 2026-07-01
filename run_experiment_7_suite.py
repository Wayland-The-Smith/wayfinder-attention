#!/usr/bin/env python3

"""
Experiment 7 — long-context retrieval benchmark suite.

Two-stage protocol (``long_context_benchmark.training_protocol: two_stage``):
  Stage A   — train ``dense_flash`` at each T → ``dense_checkpoints/T{T}_dense_flash.pt``
  Stage A.5 — train ``routing_asymmetric`` router on NIAH dense teacher → ``index_checkpoints/T{T}_routing_index.pt``
  Stage B   — all other variants load C_dense(T); routing loads task router (frozen);
              learned_address trains addresses+meat on NIAH sparse top-k LM loss.

Profiles (configs/experiment_7.yaml):
  full  — default; 6 layers, fair three-stage at T<=8192 (16k/32k deferred)
  fast  — 2 layers, T<=8192, shorter index pretrain

Step budget: run ``scripts/calibrate_dense_niah.py`` first, then pass
``--calibration-steps-json experiments/Experiment_7/dense_calibration/latest.json``.
"""



from __future__ import annotations



import argparse

import json

import sys

import traceback

from datetime import datetime, timezone

from pathlib import Path



import torch



ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))



from experiments import experiment_7

from routing_attention.benchmarks.long_context import (

    LongContextBenchmarkConfig,

    evaluate_success_criteria,

    generate_markdown_report,

    save_comparison_plots,

)

from routing_attention.benchmarks.long_context.comparison import build_comparison_table

from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram

from routing_attention.benchmarks.long_context.production_backends import (

    assert_production_backends_available,

    production_manifest_for_variants,

)

from routing_attention.benchmarks.long_context.index_pretrain import (
    pretrain_router_on_dense_checkpoint,
    router_index_checkpoint_path,
)
from routing_attention.benchmarks.long_context.dense_calibration import (
    apply_steps_to_config,
    load_calibration_recommendation,
)
from routing_attention.benchmarks.long_context.suite_profile import (

    apply_suite_profile,

    dense_checkpoint_path,

    dense_pretrain_max_context,

    estimate_run_count,

    needs_dense_checkpoint,

    resolve_dense_init_checkpoint,

    resolve_profile_name,

    variant_run_mode,

)

from routing_attention.models.fast_attention import backend_status

from routing_attention.utils.config import load_config, merge_configs

from routing_attention.utils.experiment import get_experiments_root



DEFAULT_VARIANTS = [

    "dense_flash",

    "linear",

    "local_window64",

    "local_window256",

    "routing_asymmetric",

    "learned_address_k32",

    "key_vector_k32",

]





def preflight(dry_run: bool) -> dict:

    info = collect_device_info(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    info["torch_version"] = torch.__version__

    info["cuda_version"] = torch.version.cuda

    info.update(backend_status())

    info["dry_run"] = dry_run

    print("=== Experiment 7 preflight ===")

    for k, v in info.items():

        print(f"  {k}: {v}")

    if info["device_type"] != "cuda":

        print("WARNING: CUDA not available — suite will run on CPU (slow, not representative).")

    if not info.get("fla_linear"):

        print("ERROR: flash-linear-attention (pip install flash-linear-attention) required.")

        sys.exit(1)

    if not info.get("flex_sliding_window"):

        print("ERROR: PyTorch Flex Attention required (torch>=2.4).")

        sys.exit(1)

    try:

        assert_production_backends_available()

    except RuntimeError as exc:

        print(f"ERROR: {exc}")

        sys.exit(1)

    print("Production attention backends:")

    for var, meta in production_manifest_for_variants(DEFAULT_VARIANTS).items():

        print(f"  {var}: {meta.get('kernel')} ({meta.get('package', '')})")

    print()

    return info





def _load_suite_config(profile: str | None, extra_override: dict | None) -> tuple[dict, dict]:

    cfg_path = ROOT / "configs" / "experiment_7.yaml"

    raw = load_config(cfg_path)

    profile_name = resolve_profile_name(raw, profile)

    config = apply_suite_profile(raw, profile_name)

    profile_meta = config.get("suite_active_profile", {})

    if extra_override:

        config = merge_configs(config, extra_override)

    return config, profile_meta





def _benchmark_config_from_merged(config: dict, dry_run: bool) -> LongContextBenchmarkConfig:

    bench = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))

    if dry_run:

        bench = bench.apply_dry_run_profile()

    return bench





def _load_success_criteria_config() -> dict:

    cfg_path = ROOT / "configs" / "experiment_7.yaml"

    if not cfg_path.exists():

        return {}

    return load_config(cfg_path).get("success_criteria", {})





def _config_override_for_run(

    suite_config: dict,

    extra_override: dict | None,

) -> dict:

    ovr: dict = {

        "model": dict(suite_config.get("model", {})),

        "transformer": dict(suite_config.get("transformer", {})),

        "data": dict(suite_config.get("data", {})),

        "evaluation": dict(suite_config.get("evaluation", {})),

        "long_context_benchmark": dict(suite_config.get("long_context_benchmark", {})),

        "index_pretrain": dict(suite_config.get("index_pretrain", {})),

        "router": dict(suite_config.get("router", {})),

        "data_collection": dict(suite_config.get("data_collection", {})),

        "suite_active_profile": dict(suite_config.get("suite_active_profile", {})),

    }

    if extra_override:

        ovr = merge_configs(ovr, extra_override)

    return ovr





def _enrich_entry_from_result(entry: dict, var: str, result: dict) -> None:

    var_result = result["variants"].get(var, {})

    summary = var_result.get("summary", {})

    entry["status"] = "ok"

    entry["summary"] = summary

    entry["overall_accuracy"] = summary.get("overall_accuracy")

    entry["by_task_type"] = summary.get("by_task_type", {})

    entry["by_context_length"] = summary.get("by_context_length", {})

    entry["eval_errors"] = var_result.get("eval_errors", 0)

    entry["peak_vram_mb"] = var_result.get("peak_vram_mb")

    entry["eval_latency_ms"] = var_result.get("eval_latency_ms")

    entry["tokens_per_sec"] = var_result.get("tokens_per_sec")

    entry["latency_benchmark"] = var_result.get("latency_benchmark", {})

    entry["model_device"] = var_result.get("model_device", {})

    entry["top_failures"] = var_result.get("top_failures", [])

    entry["shared_init"] = result.get("shared_init")

    entry["runtime"] = result.get("runtime", {})

    entry["attention_backends"] = result.get("attention_backends", {})

    entry["train_context_length"] = result.get("train_context_length")

    entry["suite_profile"] = result.get("suite_profile")

    if entry.get("run_mode") == "latency_only":

        entry["status"] = "latency_only"

    if entry["eval_errors"]:

        entry["status"] = "oom_or_eval_error"





def main() -> None:

    parser = argparse.ArgumentParser(description="Experiment 7 long-context benchmark suite")

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--skip-training", action="store_true")

    parser.add_argument(
        "--force-retrain-dense",
        action="store_true",
        help="Re-run Stage A dense pretrain even if checkpoint exists",
    )

    parser.add_argument(

        "--profile",

        choices=("fast", "full"),

        default=None,

        help="Suite profile (default from config: fast)",

    )

    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)

    parser.add_argument(

        "--context-lens",

        type=int,

        nargs="*",

        help="Override context lengths (longest-first sub-experiments)",

    )

    parser.add_argument(
        "--calibration-steps-json",
        type=Path,
        default=None,
        help="JSON from scripts/calibrate_dense_niah.py — sets dense/sparse step budgets",
    )

    args = parser.parse_args()



    extra_override: dict = {}

    if args.context_lens:

        extra_override["long_context_benchmark"] = {

            "context_lengths": sorted(args.context_lens, reverse=True),

        }

    if args.calibration_steps_json:
        cal = load_calibration_recommendation(args.calibration_steps_json)
        steps = int(cal["recommended_steps"])
        extra_override = merge_configs(extra_override, apply_steps_to_config({}, steps))
        print(
            f"=== Using calibrated step budget: {steps} "
            f"(verdict={cal.get('verdict')}, source={cal.get('source')}) ===\n"
        )



    suite_config, profile_meta = _load_suite_config(args.profile, extra_override or None)

    profile_name = profile_meta.get("name", args.profile or "fast")



    preflight_info = preflight(args.dry_run)

    bench_cfg = _benchmark_config_from_merged(suite_config, args.dry_run)

    context_lengths = bench_cfg.context_lengths

    success_criteria_cfg = _load_success_criteria_config()

    run_override = _config_override_for_run(suite_config, extra_override or None)



    n_planned = estimate_run_count(context_lengths, args.variants, profile_meta)

    print(f"=== Suite profile: {profile_name} ===")

    print(f"  {profile_meta.get('description', '')}")

    print(f"  n_layers={suite_config.get('model', {}).get('n_layers')}")

    print(f"  max_steps={suite_config.get('transformer', {}).get('max_steps')}")

    print(f"  context_lengths={context_lengths}")

    print(f"  planned runs={n_planned} (variants×lengths, respecting dense cap)")

    print()



    suite_dir = get_experiments_root() / "Experiment_7" / "suite_long_context"

    suite_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    tag = f"{'dry_run' if args.dry_run else 'full'}_{profile_name}"

    jsonl_path = suite_dir / f"suite_{tag}_{stamp}.jsonl"

    entries: list[dict] = []

    two_stage = (
        suite_config.get("long_context_benchmark", {}).get("training_protocol", "two_stage")
        == "two_stage"
    )
    dense_ckpt_dir = suite_dir / "dense_checkpoints"
    dense_ckpt_dir.mkdir(parents=True, exist_ok=True)
    index_ckpt_dir = suite_dir / "index_checkpoints"
    index_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if two_stage:
        print(f"Training protocol: two_stage + index_pretrain")
        print(f"  dense checkpoints -> {dense_ckpt_dir}")
        print(f"  router indices    -> {index_ckpt_dir}")

    # Stage A pre-pass (ascending T): train dense_flash so C_dense(T) exists before sparse fine-tunes.
    if two_stage and not args.skip_training:
        pretrain_cap = dense_pretrain_max_context(profile_meta)
        for train_t in sorted(context_lengths):
            if pretrain_cap is not None and train_t > pretrain_cap:
                continue
            ckpt = dense_checkpoint_path(dense_ckpt_dir, train_t)
            if ckpt.exists() and not args.force_retrain_dense:
                print(f"Stage A cached: {ckpt}")
                continue
            print(f"\n########## Stage A: dense_flash T={train_t} ##########")
            try:
                experiment_7.run(
                    variant="dense_flash",
                    dry_run=args.dry_run,
                    config_override=run_override,
                    variants=["dense_flash"],
                    train_context_length=train_t,
                    run_mode="full",
                    training_stage="dense_pretrain",
                    save_dense_checkpoint=ckpt,
                )
                print(f"Stage A saved: {ckpt}")
            except Exception:
                print(traceback.format_exc())
                print(f"ERROR: Stage A failed at T={train_t}")
                sys.exit(1)

        # Stage A.5: NIAH router index from dense teacher (routing_asymmetric only)
        index_enabled = bool(suite_config.get("index_pretrain", {}).get("enabled", True))
        if index_enabled and "routing_asymmetric" in args.variants:
            for train_t in sorted(context_lengths):
                if pretrain_cap is not None and train_t > pretrain_cap:
                    continue
                dense_ckpt = dense_checkpoint_path(dense_ckpt_dir, train_t)
                if not dense_ckpt.exists():
                    print(f"ERROR: Stage A.5 needs dense ckpt at T={train_t}")
                    sys.exit(1)
                idx_ckpt = router_index_checkpoint_path(index_ckpt_dir, train_t)
                if idx_ckpt.exists() and not args.force_retrain_dense:
                    print(f"Stage A.5 cached: {idx_ckpt}")
                    continue
                print(f"\n########## Stage A.5: router index T={train_t} ##########")
                try:
                    pretrain_router_on_dense_checkpoint(
                        suite_config,
                        dense_ckpt,
                        train_t,
                        idx_ckpt,
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        dry_run=args.dry_run,
                    )
                    print(f"Stage A.5 saved: {idx_ckpt}")
                except Exception:
                    print(traceback.format_exc())
                    print(f"ERROR: Stage A.5 failed at T={train_t}")
                    sys.exit(1)

    for train_t in context_lengths:

        print(f"\n########## Context length T={train_t} ##########")

        for var in args.variants:

            mode = variant_run_mode(profile_meta, train_t, var)

            if mode == "skip":

                print(f"SKIP T={train_t} {var} (profile)")

                continue

            dense_ckpt = resolve_dense_init_checkpoint(
                dense_ckpt_dir,
                train_t,
                profile_meta,
                context_lengths=context_lengths,
            )

            if two_stage and needs_dense_checkpoint(args.variants, profile_meta, train_t, var):
                if dense_ckpt is None or not dense_ckpt.exists():
                    entry = {
                        "train_context_length": train_t,
                        "variant": var,
                        "run_mode": mode,
                        "suite_profile": profile_name,
                        "status": "error",
                        "error": f"No dense checkpoint for T={train_t} (two-stage protocol)",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "dry_run": args.dry_run,
                    }
                    entries.append(entry)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                    print(f"ERROR: missing dense checkpoint for {var} at T={train_t}")
                    continue
                if dense_ckpt != dense_checkpoint_path(dense_ckpt_dir, train_t) and not profile_meta.get(
                    "fair_comparison", True
                ):
                    print(f"  dense init fallback: {dense_ckpt.name} -> finetune at T={train_t}")

            if var == "dense_flash":
                if mode == "full" and not args.skip_training and two_stage:
                    stage = (
                        "eval_only"
                        if dense_ckpt and dense_ckpt.exists() and not args.force_retrain_dense
                        else "dense_pretrain"
                    )
                elif mode == "latency_only":
                    stage = "eval_only"
                else:
                    stage = None
            elif two_stage and mode == "full":
                stage = "finetune_from_dense"
            else:
                stage = None

            print(f"\n=== T={train_t} variant: {var} ({mode}, stage={stage}) ===")

            router_idx = (
                router_index_checkpoint_path(index_ckpt_dir, train_t)
                if var == "routing_asymmetric"
                else None
            )

            entry: dict = {

                "train_context_length": train_t,

                "variant": var,

                "run_mode": mode,

                "training_stage": stage,

                "dense_checkpoint": str(dense_ckpt) if dense_ckpt else None,

                "router_index_checkpoint": str(router_idx) if router_idx else None,

                "suite_profile": profile_name,

                "status": "running",

                "started_at": datetime.now(timezone.utc).isoformat(),

                "dry_run": args.dry_run,

            }

            try:

                result = experiment_7.run(

                    variant=var,

                    dry_run=args.dry_run,

                    config_override=run_override,

                    variants=[var],

                    skip_training=args.skip_training,

                    train_context_length=train_t,

                    run_mode=mode,

                    training_stage=stage,

                    dense_checkpoint_path=dense_ckpt,

                    router_index_checkpoint_path=router_idx,

                    save_dense_checkpoint=(
                        dense_checkpoint_path(dense_ckpt_dir, train_t)
                        if stage == "dense_pretrain"
                        else None
                    ),

                )

                _enrich_entry_from_result(entry, var, result)

                print(

                    f"OK T={train_t} {var} mode={mode} accuracy={entry['overall_accuracy']} "

                    f"device={entry.get('model_device', {}).get('param_device')} "

                    f"peak_vram_mb={entry.get('peak_vram_mb')} "

                    f"latency_ms={entry.get('eval_latency_ms')} "

                    f"eval_errors={entry.get('eval_errors')}"

                )

            except Exception:

                entry["status"] = "error"

                entry["traceback"] = traceback.format_exc()

                print(entry["traceback"])

            entry["finished_at"] = datetime.now(timezone.utc).isoformat()

            with open(jsonl_path, "a", encoding="utf-8") as f:

                f.write(json.dumps(entry) + "\n")

            entries.append(entry)

            reset_peak_vram(torch.device("cuda" if torch.cuda.is_available() else "cpu"))



    n_ok = sum(1 for e in entries if e["status"] == "ok")

    n_lat = sum(1 for e in entries if e["status"] == "latency_only")

    n_err = sum(1 for e in entries if e["status"] == "error")

    n_oom = sum(1 for e in entries if e["status"] == "oom_or_eval_error")



    min_eval_t = int(success_criteria_cfg.get("min_context_length", 8192))
    variant_results: dict[str, dict] = {}
    for e in entries:
        if e.get("status") != "ok" or not e.get("summary"):
            continue
        if int(e.get("train_context_length", 0)) == min_eval_t:
            variant_results[e["variant"]] = e

    criteria_result = evaluate_success_criteria(variant_results, success_criteria_cfg)



    comparison_dir = suite_dir / f"comparison_{tag}_{stamp}"

    plot_paths = save_comparison_plots(entries, comparison_dir)

    comparison_table = build_comparison_table(entries)



    summary = {

        "experiment": "Experiment_7",

        "suite_profile": profile_name,

        "profile_meta": profile_meta,

        "dry_run": args.dry_run,

        "preflight": preflight_info,

        "context_lengths": context_lengths,

        "protocol": "three_stage_dense_index_sparse" if two_stage else "fixed_T_per_subexperiment",
        "two_stage": two_stage,
        "index_pretrain": bool(suite_config.get("index_pretrain", {}).get("enabled", True)),
        "dense_checkpoint_dir": str(dense_ckpt_dir),
        "index_checkpoint_dir": str(index_ckpt_dir),

        "variants_total": len(entries),

        "variants_ok": n_ok,

        "variants_latency_only": n_lat,

        "variants_error": n_err,

        "variants_oom_or_eval_error": n_oom,

        "success_criteria": success_criteria_cfg,

        "success_evaluation": criteria_result,

        "comparison_table": comparison_table,

        "comparison_plots": plot_paths,

        "runs": entries,

    }



    summary_path = suite_dir / f"suite_{tag}_summary_{stamp}.json"

    with open(summary_path, "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2)



    report_path = suite_dir / f"REPORT_{tag}_{stamp}.md"

    generate_markdown_report(summary, output_path=report_path)



    print(f"\nSuite summary: {summary_path}")

    print(f"Comparison plots: {comparison_dir}")

    print(f"Report: {report_path}")

    print(f"  ok={n_ok}  latency_only={n_lat}  oom/eval_errors={n_oom}  hard_errors={n_err}")

    print(f"  success tier: {criteria_result['tier']}")

    if args.dry_run and n_err > 0:

        sys.exit(1)





if __name__ == "__main__":

    main()


