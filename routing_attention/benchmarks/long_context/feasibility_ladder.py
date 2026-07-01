"""NIAH feasibility ladder — sequential Level 0–3 dense/variant gates."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.common import init_experiment_runtime, load_experiment_config, set_seed
from experiments.experiment_7 import (
    _build_variant_model,
    _save_dense_checkpoint,
    _train_on_benchmark,
    _verify_staged_training_protocol,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dense_calibration import analyze_holdout_curve
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import (
    LongContextSample,
    LongContextSampleGenerator,
)
from routing_attention.benchmarks.long_context.holdout import (
    clear_holdout_cache,
    resolve_holdout_splits,
)
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.config import load_config, merge_configs
from routing_attention.utils.experiment import get_experiments_root

logger = logging.getLogger("feasibility_ladder")


def load_feasibility_ladder_config(path: Path | None = None) -> dict:
    cfg_path = path or (Path(__file__).resolve().parents[3] / "configs" / "feasibility_ladder.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return raw.get("feasibility_ladder", raw)


def _build_overfit_holdout(
    bench_cfg: LongContextBenchmarkConfig,
    train_context_length: int,
) -> list[LongContextSample]:
    n = int(bench_cfg.overfit_train_samples)
    gen = LongContextSampleGenerator(bench_cfg)
    depths = gen._depths
    modes = gen._modes
    tasks = gen._tasks
    out: list[LongContextSample] = []
    for i in range(n):
        out.append(
            gen.generate_one(
                context_length=train_context_length,
                needle_depth=depths[i % len(depths)],
                task_type=tasks[i % len(tasks)],
                haystack_mode=modes[i % len(modes)],
                seed=bench_cfg.seed + i,
            )
        )
    return out


def _merge_level_config(
    ladder_cfg: dict,
    level: dict,
    *,
    dry_run: bool,
) -> dict:
    profile = ladder_cfg.get("suite_profile", "full")
    raw = load_config(Path(__file__).resolve().parents[3] / "configs" / "experiment_7.yaml")
    config = apply_suite_profile(raw, profile)
    dry = dict(ladder_cfg.get("dry_run", {}))

    bench_patch = dict(level.get("long_context_benchmark", {}))
    transformer_patch = dict(level.get("transformer", {}))
    cal_patch = dict(level.get("dense_calibration", {}))
    data_patch = dict(level.get("data", {}))

    if dry_run:
        is_variant_level = bool(level.get("variants"))
        variant_steps = int(
            dry.get(
                "level4_max_steps",
                dry.get("level3_max_steps", dry.get("max_steps", 50)),
            )
        )
        step_cap = variant_steps if is_variant_level else int(dry.get("max_steps", 50))
        transformer_patch.update(
            {
                "max_steps": step_cap,
                "dense_pretrain_steps": step_cap,
                "sparse_finetune_steps": step_cap,
                "validate_every": int(dry.get("validate_every", 10)),
                "validate_every_min": int(dry.get("validate_every", 10)),
                "log_every": int(dry.get("log_every", 10)),
            }
        )
        bench_patch["eval_samples_per_cell"] = int(dry.get("eval_samples_per_cell", 1))
        if ladder_cfg.get("holdout", {}).get("total_samples"):
            bench_patch.pop("eval_samples_per_cell", None)
        cal_patch["mid_train_samples_per_cell"] = int(dry.get("mid_train_samples_per_cell", 1))
        if dry.get("data"):
            data_patch.update(dry["data"])
        elif int(level.get("train_context_length", 0)) >= 8192:
            data_patch["batch_size"] = 1

    if ladder_cfg.get("synthetic", True):
        bench_patch.setdefault("benchmark_family", "synthetic")
    if "seed" in ladder_cfg:
        bench_patch.setdefault("seed", int(ladder_cfg["seed"]))

    ovr = {
        "holdout": {**ladder_cfg.get("holdout", {}), **level.get("holdout", {})},
        **({"seed": int(ladder_cfg["seed"])} if "seed" in ladder_cfg else {}),
        **(
            {"training": {**config.get("training", {}), **dict(ladder_cfg.get("training", {}))}}
            if ladder_cfg.get("training")
            else {}
        ),
        "model": {**config.get("model", {}), **level.get("model", {})},
        "transformer": {**config.get("transformer", {}), **transformer_patch},
        "data": {**config.get("data", {}), **data_patch},
        "long_context_benchmark": {**config.get("long_context_benchmark", {}), **bench_patch},
        "dense_calibration": {**config.get("dense_calibration", {}), **cal_patch},
        "index_pretrain": {**config.get("index_pretrain", {}), **level.get("index_pretrain", {})},
        "suite_active_profile": config.get("suite_active_profile", {}),
        "feasibility_level": {
            "level": level.get("level"),
            "name": level.get("name"),
            "description": level.get("description"),
        },
    }
    merged = merge_configs(load_experiment_config(7, variant="dense_flash"), ovr)
    bench_resolved = _resolve_bench_cfg(merged, level)
    merged["long_context_benchmark"] = bench_resolved.to_dict()
    return merged


def _resolve_bench_cfg(config: dict, level: dict) -> LongContextBenchmarkConfig:
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    if bench_cfg.benchmark_family == "synthetic":
        bench_cfg = bench_cfg.apply_synthetic_profile()
    train_t = int(level["train_context_length"])
    if train_t not in bench_cfg.context_lengths:
        lengths = sorted(set(bench_cfg.context_lengths + [train_t]), reverse=True)
        bench_cfg = LongContextBenchmarkConfig.from_dict(
            {**bench_cfg.to_dict(), "context_lengths": lengths}
        )
        if bench_cfg.benchmark_family == "synthetic":
            bench_cfg = bench_cfg.apply_synthetic_profile()
    return bench_cfg


def evaluate_pass_criteria(level: dict, final_eval: dict) -> dict[str, Any]:
    criteria = dict(level.get("pass_criteria", {}))
    passed = True
    reasons: list[str] = []
    by_task = final_eval.get("by_task_type") or {}

    min_gate = criteria.get("primary_gate_accuracy_min")
    if min_gate is not None:
        acc = float(
            final_eval.get(
                "primary_gate_accuracy",
                final_eval.get("pure_niah_accuracy", final_eval.get("overall_accuracy", 0.0)),
            )
        )
        if acc < float(min_gate):
            passed = False
            reasons.append(f"primary_gate {acc:.2%} < {float(min_gate):.0%}")

    for task, thresh in (criteria.get("by_task_type") or {}).items():
        acc = by_task.get(task)
        if acc is None:
            passed = False
            reasons.append(f"missing task metric: {task}")
        elif float(acc) < float(thresh):
            passed = False
            reasons.append(f"{task} {float(acc):.2%} < {float(thresh):.0%}")

    return {
        "passed": passed,
        "reasons": reasons,
        "criteria": criteria,
    }


def run_dense_level(
    ladder_cfg: dict,
    level: dict,
    *,
    dry_run: bool,
    output_dir: Path,
    device_info: dict | None = None,
    dense_checkpoint: Path | None = None,
) -> dict[str, Any]:
    level_id = int(level["level"])
    train_t = int(level["train_context_length"])
    t_wall_start = time.perf_counter()

    config = _merge_level_config(ladder_cfg, level, dry_run=dry_run)
    bench_cfg = _resolve_bench_cfg(config, level)

    device = init_experiment_runtime(config)
    set_seed(config.get("seed", 45))
    if device_info is None:
        device_info = collect_device_info(device)
        device_info.update(backend_status())

    clear_holdout_cache()
    bench_level = dict(level.get("long_context_benchmark", {}))
    if bench_level.get("overfit_eval_same_samples") and bench_cfg.overfit_train_samples > 0:
        holdout_full = _build_overfit_holdout(bench_cfg, train_t)
        holdout_mid = holdout_full
        holdout_meta = {
            "holdout_full_samples": len(holdout_full),
            "holdout_mid_samples": len(holdout_mid),
            "holdout_total_target": None,
            "overfit_eval_same_samples": True,
        }
    else:
        holdout_mid, holdout_full, holdout_meta = resolve_holdout_splits(
            config,
            bench_cfg,
            train_t,
            mid_train_seed_offset=level_id,
        )
    if not holdout_full:
        raise RuntimeError(f"Level {level_id}: no holdout samples for T={train_t}")

    train_steps = int(config.get("transformer", {}).get("max_steps", 0))
    validate_every = int(config.get("transformer", {}).get("validate_every", 0))
    cal_cfg = config.get("dense_calibration", {})

    print(f"\n=== Level {level_id}: {level.get('name')} ===")
    print(f"  {level.get('description', '')}")
    print(f"  T={train_t}  layers={config.get('model', {}).get('n_layers')}")
    print(f"  tasks={bench_cfg.task_types}")
    print(f"  steps={train_steps}  validate_every={validate_every}")
    print(f"  holdout_mid={len(holdout_mid)}  holdout_official={len(holdout_full)}")
    if bench_cfg.context_curriculum:
        print(f"  curriculum={bench_cfg.context_curriculum}")
    if bench_cfg.suffix_curriculum:
        print(f"  suffix_curriculum={bench_cfg.suffix_curriculum}")
    if bench_cfg.overfit_train_samples:
        print(f"  overfit_train_samples={bench_cfg.overfit_train_samples}")
    if dense_checkpoint is not None:
        print(f"  dense_checkpoint={dense_checkpoint}")
    print()

    reset_peak_vram(device)
    model, var_config = _build_variant_model(
        config,
        "dense_flash",
        device,
        train_t,
        dense_checkpoint=dense_checkpoint,
    )
    audit = _verify_staged_training_protocol(
        "dense_flash",
        "dense_pretrain",
        getattr(model, "_exp7_routing_info", {}),
        model,
        two_stage=True,
    )

    train_info = _train_on_benchmark(
        model,
        var_config,
        bench_cfg,
        holdout_mid,
        device,
        train_t,
        logger,
        max_steps=train_steps,
        training_stage="dense_pretrain",
    )

    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout_full)
    eval_t0 = time.perf_counter()
    final_summary = evaluator.evaluate_module(model, device=device, show_progress=True)
    eval_wall_sec = time.perf_counter() - eval_t0
    wall_sec = time.perf_counter() - t_wall_start
    peak_mb = peak_vram_mb(device)

    recommendation = analyze_holdout_curve(
        train_info.get("mid_validations", []),
        min_delta_pp=float(cal_cfg.get("min_delta_pp", 0.005)),
        patience_checks=int(cal_cfg.get("patience_checks", 3)),
        target_accuracy=cal_cfg.get("target_accuracy"),
        min_recommended_steps=int(cal_cfg.get("min_recommended_steps", 1000)),
    )
    pass_result = evaluate_pass_criteria(level, final_summary.to_dict())

    checkpoint_path = None
    if level.get("save_checkpoint"):
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"level{level_id}_T{train_t}_dense_flash.pt"
        _save_dense_checkpoint(
            model,
            checkpoint_path,
            train_context_length=train_t,
            trained_steps=int(train_info.get("trained_steps", 0)),
        )

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "feasibility_ladder_level",
        "level": level_id,
        "name": level.get("name"),
        "description": level.get("description"),
        "dry_run": dry_run,
        "train_context_length": train_t,
        "n_layers": config.get("model", {}).get("n_layers"),
        "benchmark_family": bench_cfg.benchmark_family,
        "benchmark_version": bench_cfg.benchmark_version,
        "task_types": list(bench_cfg.task_types),
        "wall_sec": wall_sec,
        "eval_wall_sec": eval_wall_sec,
        "peak_vram_mb": peak_mb,
        "device_info": device_info,
        "training": train_info,
        "final_eval": final_summary.to_dict(),
        "recommendation": recommendation,
        "pass_result": pass_result,
        "staged_training_audit": audit,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "dense_init_checkpoint": str(dense_checkpoint) if dense_checkpoint else None,
        "benchmark_config": bench_cfg.to_dict(),
    }
    return payload


def run_variant_level(
    ladder_cfg: dict,
    level: dict,
    *,
    dry_run: bool,
    output_dir: Path,
    dense_checkpoint: Path,
    device_info: dict | None = None,
) -> dict[str, Any]:
    from experiments import experiment_7
    from routing_attention.benchmarks.long_context.index_pretrain import (
        pretrain_router_on_dense_checkpoint,
        router_index_checkpoint_path,
    )

    level_id = int(level["level"])
    train_t = int(level["train_context_length"])
    dry = dict(ladder_cfg.get("dry_run", {}))
    variants = list(
        dry.get("level4_variants")
        or dry.get("level3_variants")
        if dry_run and (dry.get("level4_variants") or dry.get("level3_variants"))
        else level.get("variants", ["dense_flash"])
    )

    config = _merge_level_config(ladder_cfg, level, dry_run=dry_run)
    if dry_run and dry.get("skip_index_pretrain"):
        config.setdefault("index_pretrain", {})["enabled"] = False

    t0 = time.perf_counter()
    print(f"\n=== Level {level_id}: {level.get('name')} ===")
    print(f"  variants={variants}")
    print(f"  dense_checkpoint={dense_checkpoint}")
    print()

    index_ckpt_dir = output_dir / "index_checkpoints"
    index_ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (
        not dry_run
        and config.get("index_pretrain", {}).get("enabled", True)
        and "routing_asymmetric" in variants
        and dense_checkpoint.exists()
    ):
        idx_ckpt = router_index_checkpoint_path(index_ckpt_dir, train_t)
        if not idx_ckpt.exists():
            pretrain_router_on_dense_checkpoint(
                config,
                dense_checkpoint,
                train_t,
                idx_ckpt,
                device,
                dry_run=dry_run,
            )

    variant_results: dict[str, Any] = {}
    for var in variants:
        var_t0 = time.perf_counter()
        print(f"--- Level {level_id} variant: {var} ---")
        idx_ckpt = router_index_checkpoint_path(index_ckpt_dir, train_t)
        router_ckpt = idx_ckpt if var == "routing_asymmetric" and idx_ckpt.exists() else None
        try:
            # Steps are already capped in config_override; avoid apply_dry_run_profile
            # which would switch synthetic → NL tasks.
            result = experiment_7.run(
                variant=var,
                dry_run=False,
                config_override=config,
                variants=[var],
                train_context_length=train_t,
                run_mode="full",
                training_stage="finetune_from_dense" if var != "dense_flash" else "eval_only",
                dense_checkpoint_path=dense_checkpoint,
                router_index_checkpoint_path=router_ckpt,
            )
            var_result = result.get("variants", {}).get(var, {})
            variant_results[var] = {
                "status": "ok",
                "wall_sec": time.perf_counter() - var_t0,
                "summary": var_result.get("summary", {}),
                "peak_vram_mb": var_result.get("peak_vram_mb"),
                "eval_latency_ms": var_result.get("eval_latency_ms"),
                "tokens_per_sec": var_result.get("tokens_per_sec"),
                "latency_benchmark": var_result.get("latency_benchmark", {}),
            }
        except Exception as exc:
            variant_results[var] = {
                "status": "error",
                "error": str(exc),
                "wall_sec": time.perf_counter() - var_t0,
            }

    wall_sec = time.perf_counter() - t0
    if device_info is None:
        device_info = collect_device_info(device)
        device_info.update(backend_status())

    dense_summary = (variant_results.get("dense_flash") or {}).get("summary") or {}
    local_summary = (variant_results.get("local_window64") or {}).get("summary") or {}
    routing_summary = (variant_results.get("routing_asymmetric") or {}).get("summary") or {}

    def _gate(s: dict) -> float | None:
        if not s:
            return None
        return s.get("primary_gate_accuracy", s.get("pure_niah_accuracy", s.get("overall_accuracy")))

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "feasibility_ladder_level",
        "level": level_id,
        "name": level.get("name"),
        "dry_run": dry_run,
        "train_context_length": train_t,
        "variants": variants,
        "wall_sec": wall_sec,
        "device_info": device_info,
        "variant_results": variant_results,
        "comparison": {
            "dense_gate": _gate(dense_summary),
            "local_window64_gate": _gate(local_summary),
            "routing_asymmetric_gate": _gate(routing_summary),
        },
        "pass_result": {
            "passed": True,
            "reasons": [],
            "note": "Level 3 is informational unless explicit criteria are set",
        },
    }
    return payload


def run_feasibility_ladder(
    *,
    dry_run: bool = False,
    levels: list[int] | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    stop_on_failure: bool | None = None,
) -> dict[str, Any]:
    ladder_cfg = load_feasibility_ladder_config(config_path)
    if stop_on_failure is None:
        stop_on_failure = bool(ladder_cfg.get("stop_on_failure", True))

    out_dir = output_dir or (get_experiments_root() / "Experiment_7" / "feasibility_ladder")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = collect_device_info(device)
    device_info.update(backend_status())
    if device.type == "cuda":
        try:
            assert_production_backends_available()
        except RuntimeError as exc:
            if not dry_run:
                raise
            device_info["backend_warning"] = str(exc)

    all_levels = list(ladder_cfg.get("levels", []))
    if levels is not None:
        wanted = set(levels)
        all_levels = [lv for lv in all_levels if int(lv.get("level", -1)) in wanted]

    suite_t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    checkpoints: dict[int, Path] = {}
    aborted = False
    abort_reason = ""

    for level in all_levels:
        level_id = int(level["level"])
        train_t = int(level["train_context_length"])
        requires = level.get("requires_level")
        dense_ckpt: Path | None = None

        if requires is not None:
            req_id = int(requires)
            dense_ckpt = checkpoints.get(req_id)
            if dense_ckpt is None or not dense_ckpt.exists():
                dense_ckpt = out_dir / "checkpoints" / f"level{req_id}_T{train_t}_dense_flash.pt"
            if dense_ckpt.exists():
                checkpoints[req_id] = dense_ckpt

        if requires is not None and req_id not in checkpoints:
            prev = next((r for r in results if r.get("level") == req_id), None)
            if (
                prev
                and not prev.get("pass_result", {}).get("passed", False)
                and stop_on_failure
                and not dry_run
            ):
                aborted = True
                abort_reason = f"Level {requires} did not pass"
                results.append(
                    {
                        "level": level_id,
                        "name": level.get("name"),
                        "status": "skipped",
                        "reason": abort_reason,
                    }
                )
                break
            if dense_ckpt is None or not dense_ckpt.exists():
                if level.get("variants") and dry_run:
                    req = int(requires or 3)
                    dense_ckpt = out_dir / "checkpoints" / f"level{req}_T{train_t}_dense_flash.pt"
                if dense_ckpt is None or not dense_ckpt.exists():
                    aborted = True
                    abort_reason = f"Missing checkpoint from level {requires}"
                    results.append(
                        {
                            "level": level_id,
                            "status": "skipped",
                            "reason": abort_reason,
                        }
                    )
                    break

        try:
            if level.get("variants"):
                req = int(requires or 3)
                ckpt = checkpoints.get(req)
                if ckpt is None:
                    ckpt = out_dir / "checkpoints" / f"level{req}_T{train_t}_dense_flash.pt"
                payload = run_variant_level(
                    ladder_cfg,
                    level,
                    dry_run=dry_run,
                    output_dir=out_dir,
                    dense_checkpoint=Path(ckpt),
                    device_info=device_info,
                )
            else:
                payload = run_dense_level(
                    ladder_cfg,
                    level,
                    dry_run=dry_run,
                    output_dir=out_dir,
                    device_info=device_info,
                    dense_checkpoint=dense_ckpt if requires is not None else None,
                )
                if payload.get("checkpoint_path"):
                    checkpoints[level_id] = Path(payload["checkpoint_path"])

            results.append(payload)
            _write_level_artifact(out_dir, payload)

            passed = payload.get("pass_result", {}).get("passed", True)
            if not passed and stop_on_failure and not dry_run:
                aborted = True
                abort_reason = "; ".join(payload.get("pass_result", {}).get("reasons", []))
                break
        except Exception as exc:
            results.append(
                {
                    "level": level_id,
                    "name": level.get("name"),
                    "status": "error",
                    "error": str(exc),
                }
            )
            aborted = True
            abort_reason = str(exc)
            if stop_on_failure:
                break

    suite_wall = time.perf_counter() - suite_t0
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "feasibility_ladder_suite",
        "dry_run": dry_run,
        "wall_sec": suite_wall,
        "device_info": device_info,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "levels_run": len(results),
        "levels": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    summary_path = out_dir / f"feasibility_ladder_{tag}_{stamp}.json"
    latest_path = out_dir / "latest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_suite_summary(summary, summary_path, latest_path)
    return summary


def _write_level_artifact(output_dir: Path, payload: dict[str, Any]) -> None:
    level = payload.get("level", "?")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"level_{level}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_suite_summary(summary: dict, summary_path: Path, latest_path: Path) -> None:
    print()
    print("=== Feasibility ladder suite summary ===")
    print(f"  dry_run={summary.get('dry_run')}  wall={summary.get('wall_sec', 0):.1f}s")
    if summary.get("aborted"):
        print(f"  ABORTED: {summary.get('abort_reason')}")
    for entry in summary.get("levels", []):
        lid = entry.get("level")
        name = entry.get("name", entry.get("status", "?"))
        if entry.get("status") in ("skipped", "error"):
            print(f"  Level {lid} ({name}): {entry.get('status')} — {entry.get('reason') or entry.get('error')}")
            continue
        final = entry.get("final_eval") or {}
        gate = final.get("primary_gate_accuracy", final.get("overall_accuracy"))
        passed = entry.get("pass_result", {}).get("passed")
        print(
            f"  Level {lid} ({name}): gate={_pct(gate)} passed={passed} "
            f"wall={entry.get('wall_sec', 0):.1f}s vram={entry.get('peak_vram_mb')}MB"
        )
        if entry.get("variant_results"):
            for var, vr in entry["variant_results"].items():
                s = vr.get("summary") or {}
                acc = s.get("primary_gate_accuracy", s.get("overall_accuracy"))
                lat = vr.get("eval_latency_ms")
                print(f"    {var}: gate={_pct(acc)} status={vr.get('status')} latency={lat}ms")
    print()
    print(f"  Wrote: {summary_path}")
    print(f"  Latest: {latest_path}")


def _pct(v) -> str:
    if v is None:
        return "n/a"
    return f"{float(v) * 100:.2f}%"
