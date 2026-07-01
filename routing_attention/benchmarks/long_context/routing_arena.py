"""Routing attention arena — T=2048 pointer_unique fair comparison helpers."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.common import init_experiment_runtime, load_experiment_config, set_seed
from experiments.experiment_7 import (
    _benchmark_config_from_run,
    _build_variant_model,
    _train_on_benchmark,
    _verify_staged_training_protocol,
)
from routing_attention.benchmarks.long_context.config import (
    LongContextBenchmarkConfig,
    apply_synthetic_family_profile,
)
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.holdout import (
    resolve_holdout_splits,
)
from routing_attention.benchmarks.long_context.index_pretrain import (
    address_index_checkpoint_path,
    pretrain_addresses_on_dense_checkpoint,
)
from routing_attention.benchmarks.long_context.production_backends import (
    EXP7_PRODUCTION_BACKENDS,
    production_manifest_for_variants,
)
from routing_attention.benchmarks.long_context.routing_setup import apply_routing_variant_settings
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile
from routing_attention.models.learned_address import attach_address_book_to_model, ensure_address_book_on_model
from routing_attention.utils.checkpoint import load_checkpoint
from routing_attention.utils.config import merge_configs

logger = logging.getLogger("routing_arena")

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARENA_CONFIG = ROOT / "configs" / "routing_arena_t2048.yaml"

ROUTING_VARIANTS = ("key_vector_k32", "learned_address_k32")
BASELINE_ATTENTION_VARIANTS = ("linear", "local_window64", "local_window256")
ARENA_VARIANTS = ("dense_flash", *ROUTING_VARIANTS, *BASELINE_ATTENTION_VARIANTS)

_BASELINE_EXPECTED_ATTN: dict[str, tuple[str, int | None]] = {
    "linear": ("LinearAttention", None),
    "local_window64": ("LocalAttention", 64),
    "local_window256": ("LocalAttention", 256),
}


def load_routing_arena_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_ARENA_CONFIG
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return raw.get("routing_arena", raw)


def resolve_dense_checkpoint(
    train_t: int,
    *,
    n_layers: int = 6,
    explicit: str | Path | None = None,
) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Dense checkpoint not found: {p}")
        return p

    candidates = [
        ROOT / "experiments" / "Experiment_7" / "feasibility_ladder" / "checkpoints" / f"level2_T{train_t}_dense_flash.pt",
        ROOT / "experiments" / "Experiment_7" / "feasibility_ladder_4L_20k" / "checkpoints" / f"level2_T{train_t}_dense_flash.pt",
        ROOT / "experiments" / "Experiment_7" / "feasibility_ladder_4L" / "checkpoints" / f"level2_T{train_t}_dense_flash.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No dense checkpoint for T={train_t} (n_layers={n_layers}). "
        f"Run feasibility L2 first or pass --dense-checkpoint. Tried: {candidates}"
    )


def build_arena_experiment_config(
    arena_cfg: dict,
    *,
    dry_run: bool = False,
    n_layers: int | None = None,
) -> dict:
    """Merge arena YAML into a full experiment_7 config with synthetic pointer_unique grid."""
    profile = arena_cfg.get("suite_profile", "full")
    raw = load_experiment_config(7, variant="dense_flash")
    config = apply_suite_profile(raw, profile)

    layers = int(n_layers or arena_cfg.get("n_layers", config.get("model", {}).get("n_layers", 6)))
    train_t = int(arena_cfg["train_context_length"])
    dry = dict(arena_cfg.get("dry_run", {}))

    bench_patch = dict(arena_cfg.get("long_context_benchmark", {}))
    transformer_patch = dict(arena_cfg.get("transformer", {}))
    index_patch = dict(arena_cfg.get("index_pretrain", {}))
    la_patch = dict(arena_cfg.get("learned_address", {}))
    kv_patch = dict(arena_cfg.get("key_vector", {}))
    addr_index_steps = int(
        la_patch.get("address_index_steps")
        or index_patch.get("address_index_steps")
        or index_patch.get("router_max_steps", 2000)
    )
    index_patch["address_index_steps"] = addr_index_steps

    if dry_run:
        step_cap = int(dry.get("sparse_finetune_steps", dry.get("max_steps", 40)))
        transformer_patch.update(
            {
                "sparse_finetune_steps": step_cap,
                "max_steps": int(dry.get("max_steps", step_cap)),
                "validate_every": int(dry.get("validate_every", 10)),
                "validate_every_min": int(dry.get("validate_every", 10)),
                "log_every": int(dry.get("log_every", 10)),
            }
        )
        if arena_cfg.get("feasibility_parity", False):
            cap = int(dry.get("max_steps", step_cap))
            transformer_patch["dense_pretrain_steps"] = cap
            transformer_patch["max_steps"] = cap
        index_patch.update(
            {
                "address_index_steps": int(dry.get("address_index_steps", 50)),
            }
        )
        la_patch.update(
            {
                "address_index_steps": int(dry.get("address_index_steps", 50)),
                "joint_finetune_steps": int(dry.get("joint_finetune_steps", 40)),
            }
        )
        kv_patch.update(
            {
                "sparse_finetune_steps": int(dry.get("sparse_finetune_steps", 40)),
            }
        )
        config.setdefault("data", {})["batch_size"] = 1
        if dry.get("needle_scatter_curriculum_spec"):
            bench_patch["needle_scatter_curriculum_spec"] = dict(
                dry["needle_scatter_curriculum_spec"]
            )
            bench_patch.pop("needle_scatter_curriculum", None)

    holdout_patch = dict(arena_cfg.get("holdout", {}))
    if dry_run and dry.get("mid_train_samples_per_cell") is not None:
        holdout_patch["mid_train_samples_per_cell"] = int(dry["mid_train_samples_per_cell"])

    cal_patch = {
        **config.get("dense_calibration", {}),
        **arena_cfg.get("dense_calibration", {}),
        "live_metrics": True,
        "early_stop": False,
    }
    cal_patch.setdefault("eval_use_full_holdout", False)
    cal_patch.setdefault("restore_best_checkpoint", True)

    if "seed" in arena_cfg:
        bench_patch.setdefault("seed", int(arena_cfg["seed"]))

    ovr = {
        "holdout": holdout_patch,
        "dense_calibration": cal_patch,
        **({"seed": int(arena_cfg["seed"])} if "seed" in arena_cfg else {}),
        **(
            {"training": {**config.get("training", {}), **dict(arena_cfg.get("training", {}))}}
            if arena_cfg.get("training")
            else {}
        ),
        "model": {
            **config.get("model", {}),
            **arena_cfg.get("model", {}),
            "n_layers": layers,
            "max_seq_len": max(train_t, int(config.get("model", {}).get("max_seq_len", train_t))),
        },
        "data": {
            **config.get("data", {}),
            **dict(arena_cfg.get("data", {})),
            "dataset": "long_context",
            "seq_len": train_t,
            "train_context_length": train_t,
        },
        "transformer": {**config.get("transformer", {}), **transformer_patch},
        "long_context_benchmark": {
            **config.get("long_context_benchmark", {}),
            **bench_patch,
        },
        "index_pretrain": {**config.get("index_pretrain", {}), **index_patch},
        "routing_attention": {
            **config.get("routing_attention", {}),
            **arena_cfg.get("routing_attention", {}),
        },
        "learned_address": {
            **config.get("learned_address", {}),
            **la_patch,
        },
        "key_vector": {
            **config.get("key_vector", {}),
            **kv_patch,
        },
        "routing_arena": {
            "train_context_length": train_t,
            "dry_run": dry_run,
            "learned_address_schedule": la_patch,
            **{k: v for k, v in arena_cfg.items() if k in ("feasibility_parity", "holdout_mid_seed_offset")},
        },
    }
    merged = merge_configs(config, ovr)
    if arena_cfg.get("feasibility_parity", False):
        train_steps = int(
            transformer_patch.get("max_steps")
            or transformer_patch.get("dense_pretrain_steps")
            or transformer_patch.get("sparse_finetune_steps")
            or merged.get("transformer", {}).get("max_steps", 0)
        )
        if train_steps > 0:
            merged.setdefault("transformer", {})
            merged["transformer"]["max_steps"] = train_steps
            merged["transformer"]["dense_pretrain_steps"] = train_steps
    bench = _resolve_synthetic_bench_cfg(merged, train_t)
    bench_dict = bench.to_dict()
    for key in ("needle_scatter_curriculum_spec",):
        if key in bench_patch:
            bench_dict[key] = bench_patch[key]
    merged["long_context_benchmark"] = bench_dict
    if list(bench.task_types or []) == ["slot_pointer"]:
        merged.setdefault("model", {})
        merged["model"]["vocab_size"] = bench.vocab_size
        output_head = str(merged["model"].get("output_head", "pointer_mlp"))
        if output_head in ("lm_token", "pool_mlp_token"):
            merged["long_context_benchmark"].setdefault(
                "train_label_mode",
                bench.train_label_mode or "query_only_answer",
            )
            if output_head == "pool_mlp_token":
                merged["model"].setdefault("pool_mlp_positions", 16)
        else:
            merged["model"].setdefault("output_head", "pointer_mlp")
            merged["model"].setdefault("pointer_target_mode", "value_slots")
            merged["model"].setdefault("pointer_mlp_hidden", 2100)
            merged["model"].setdefault("num_pointer_slots", bench.num_slot_quads)
    elif list(bench.task_types or []) == ["pointer_unique"]:
        merged.setdefault("model", {})
        merged["model"].setdefault("output_head", "lm_token")
        # Feasibility ladder keeps base model vocab (257); only shrink for non-parity arena runs.
        if not arena_cfg.get("feasibility_parity", False):
            merged["model"]["vocab_size"] = bench.vocab_size
    elif list(bench.task_types or []) == ["pointer_conflict_first"]:
        merged.setdefault("model", {})
        merged["model"].setdefault("output_head", "lm_token")
        merged["model"]["vocab_size"] = bench.vocab_size
    elif list(bench.task_types or []) == ["mqar_addr_val"]:
        merged.setdefault("model", {})
        merged["model"].setdefault("output_head", "lm_token")
        merged["model"]["vocab_size"] = bench.vocab_size
        merged["long_context_benchmark"].setdefault(
            "train_label_mode",
            bench.train_label_mode or "query_only_answer",
        )
    elif list(bench.task_types or []) == ["addr_val"]:
        merged.setdefault("model", {})
        merged["model"].setdefault("output_head", "lm_token")
        merged["model"]["vocab_size"] = bench.vocab_size
        if bench.train_label_mode == "query_only_answer":
            merged["long_context_benchmark"].setdefault("train_label_mode", "query_only_answer")
    return merged


def _apply_benchmark_profile(bench_cfg: LongContextBenchmarkConfig) -> LongContextBenchmarkConfig:
    return apply_synthetic_family_profile(bench_cfg)


def _resolve_synthetic_bench_cfg(config: dict, train_t: int) -> LongContextBenchmarkConfig:
    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    bench_cfg = _apply_benchmark_profile(bench_cfg)
    if train_t not in bench_cfg.context_lengths:
        lengths = sorted(set(bench_cfg.context_lengths + [train_t]), reverse=True)
        bench_cfg = LongContextBenchmarkConfig.from_dict(
            {**bench_cfg.to_dict(), "context_lengths": lengths}
        )
        bench_cfg = _apply_benchmark_profile(bench_cfg)
    return bench_cfg


def _arena_holdout_splits(
    config: dict,
    train_t: int,
    *,
    mid_train_seed_offset: int | None = None,
) -> tuple[LongContextBenchmarkConfig, list, list, dict[str, Any]]:
    bench_cfg = _resolve_synthetic_bench_cfg(config, train_t)
    arena_cfg = config.get("routing_arena", {})
    offset = (
        int(mid_train_seed_offset)
        if mid_train_seed_offset is not None
        else int(arena_cfg.get("holdout_mid_seed_offset", 7))
    )
    holdout_mid, holdout_full, meta = resolve_holdout_splits(
        config,
        bench_cfg,
        train_t,
        mid_train_seed_offset=offset,
    )
    return bench_cfg, holdout_mid, holdout_full, meta


def _official_eval(
    model: torch.nn.Module,
    bench_cfg: LongContextBenchmarkConfig,
    holdout_full: list,
    device: torch.device,
) -> dict[str, Any]:
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout_full)
    summary = evaluator.evaluate_module(model, device=device, show_progress=True)
    out = summary.to_dict()
    out["eval_subset"] = "official_full_holdout"
    out["holdout_samples"] = len(holdout_full)
    return out


def _resolve_official_eval(
    config: dict,
    model: torch.nn.Module,
    bench_cfg: LongContextBenchmarkConfig,
    holdout_full: list,
    device: torch.device,
    train_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Dry-run skips the full holdout grid; full runs evaluate all official samples."""
    if bool(config.get("routing_arena", {}).get("dry_run")):
        bh = (train_info or {}).get("best_holdout") or {}
        acc = bh.get("primary_gate_accuracy")
        return {
            "eval_subset": "dry_run_mid_holdout_only",
            "skipped_official_full_holdout": True,
            "holdout_samples": len(holdout_full),
            "primary_gate_accuracy": acc,
            "primary_gate_correct": None,
            "primary_gate_total": None,
            "best_holdout_step": bh.get("step"),
        }
    return _official_eval(model, bench_cfg, holdout_full, device)


def run_dense_flash_finetune(
    config: dict,
    *,
    train_t: int,
    dense_ckpt: Path | None,
    device: torch.device,
    log,
    save_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune dense Flash/SDPA trunk on the arena task (same protocol as linear baseline)."""
    bench_cfg, holdout_mid, holdout_full, holdout_meta = _arena_holdout_splits(config, train_t)
    from_scratch = dense_ckpt is None
    tx = config.get("transformer", {})
    if from_scratch:
        finetune_steps = int(
            tx.get("dense_pretrain_steps") or tx.get("max_steps") or tx.get("sparse_finetune_steps") or 0
        )
    else:
        finetune_steps = int(tx.get("sparse_finetune_steps") or tx.get("max_steps") or 0)
    finetune_lr = float(config.get("transformer", {}).get("lr", 3e-4))
    backend = production_manifest_for_variants(["dense_flash"]).get("dense_flash", {})

    stage_label = "dense_pretrain" if from_scratch else "finetune_from_dense"
    print(f"\n=== dense_flash {'Stage A' if from_scratch else 'Stage B'}: "
          f"{'train' if from_scratch else 'fine-tune'} T={train_t} ===")
    if from_scratch:
        print("  dense_checkpoint=None (train from scratch)")
    else:
        print(f"  dense_checkpoint={dense_ckpt}")
    print(f"  kernel={backend.get('kernel')}  package={backend.get('package', '')}")
    print(
        f"  holdout_mid={holdout_meta['holdout_mid_samples']}  "
        f"holdout_official={holdout_meta['holdout_full_samples']}"
    )
    print(f"  steps={finetune_steps} @ lr={finetune_lr} (full trunk)")

    model, var_config = _build_variant_model(
        config,
        "dense_flash",
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    routing_info = getattr(model, "_exp7_routing_info", {})
    n_dense = sum(
        1
        for b in model.blocks
        if type(b.attn).__name__ in ("DenseSDPAAttention", "DenseAttention")
    )
    if n_dense != model.n_layers:
        raise RuntimeError(
            f"dense_flash: expected dense attention at all {model.n_layers} layers, got {n_dense}"
        )

    audit = _verify_staged_training_protocol(
        "dense_flash",
        stage_label,
        routing_info,
        model,
        two_stage=True,
    )
    print(f"  audit: {audit}")

    finetune_config = deepcopy(var_config)
    finetune_config.setdefault("transformer", {})["lr"] = finetune_lr

    train_info = _train_on_benchmark(
        model,
        finetune_config,
        bench_cfg,
        holdout_mid,
        device,
        train_t,
        log,
        max_steps=finetune_steps,
        training_stage=stage_label,
    )

    eval_official = _resolve_official_eval(
        config, model, bench_cfg, holdout_full, device, train_info
    )
    saved_checkpoint: str | None = None
    if save_checkpoint_path is not None:
        from experiments.experiment_7 import _save_dense_checkpoint

        saved_checkpoint = _save_dense_checkpoint(
            model,
            Path(save_checkpoint_path),
            train_context_length=train_t,
            trained_steps=int(train_info.get("trained_steps", 0)),
        )
        print(f"  saved dense checkpoint: {saved_checkpoint}")

    return {
        "variant": "dense_flash",
        "holdout": holdout_meta,
        "training_audit": audit,
        "train_info": train_info,
        "eval": eval_official,
        "eval_official": eval_official,
        "routing_info": routing_info,
        "production_backend": backend,
        "saved_dense_checkpoint": saved_checkpoint,
        "schedule": {
            "finetune_steps": finetune_steps,
            "finetune_lr": finetune_lr,
            "index_pretrain": None,
            "from_scratch": from_scratch,
        },
    }


def run_dense_flash_eval_from_checkpoint(
    config: dict,
    *,
    train_t: int,
    dense_ckpt: Path,
    device: torch.device,
    log,
) -> dict[str, Any]:
    """Official holdout eval for a saved dense_flash checkpoint (no training)."""
    bench_cfg, _, holdout_full, holdout_meta = _arena_holdout_splits(config, train_t)
    if not dense_ckpt.exists():
        raise FileNotFoundError(f"Dense checkpoint not found: {dense_ckpt}")

    print(f"\n=== dense_flash eval-only T={train_t} ===")
    print(f"  dense_checkpoint={dense_ckpt}")
    print(f"  holdout_official={holdout_meta['holdout_full_samples']}")

    model, _ = _build_variant_model(
        config,
        "dense_flash",
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    eval_official = _official_eval(model, bench_cfg, holdout_full, device)
    acc = eval_official.get("primary_gate_accuracy", eval_official.get("overall_accuracy"))
    acc_s = f"{float(acc) * 100:.2f}%" if acc is not None else "n/a"
    print(f"  dense official acc: {acc_s}")
    return {
        "variant": "dense_flash",
        "holdout": holdout_meta,
        "eval": eval_official,
        "eval_official": eval_official,
        "loaded_dense_checkpoint": str(dense_ckpt),
        "train_info": {"trained_steps": 0, "eval_only": True},
    }


def run_dense_flash_baseline(
    config: dict,
    *,
    train_t: int,
    dense_ckpt: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Official eval on the full standardized holdout (dense checkpoint, no training)."""
    bench_cfg, _holdout_mid, holdout_full, holdout_meta = _arena_holdout_splits(config, train_t)

    print(f"\n=== dense_flash: official holdout eval T={train_t} ===")
    print(f"  checkpoint={dense_ckpt}")
    print(
        f"  holdout_official={holdout_meta['holdout_full_samples']} "
        f"(target={holdout_meta.get('holdout_total_target')}, "
        f"{holdout_meta['eval_samples_per_cell']}/cell)"
    )

    model, var_config = _build_variant_model(
        config,
        "dense_flash",
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    audit = _verify_staged_training_protocol(
        "dense_flash",
        "eval_only",
        getattr(model, "_exp7_routing_info", {}),
        model,
        two_stage=True,
    )
    print(f"  audit: {audit}")

    eval_official = _resolve_official_eval(
        config, model, bench_cfg, holdout_full, device, train_info
    )
    return {
        "variant": "dense_flash",
        "training_audit": audit,
        "holdout": holdout_meta,
        "eval": eval_official,
        "eval_official": eval_official,
        "train_info": {"trained_steps": 0, "training_stage": "eval_only_baseline"},
        "routing_info": getattr(model, "_exp7_routing_info", {}),
    }


def run_address_index_pretrain(
    config: dict,
    dense_ckpt: Path,
    train_t: int,
    index_dir: Path,
    device: torch.device,
    *,
    dry_run: bool = False,
    force_refresh: bool = False,
    train_cache: Path | str | None = None,
    holdout_cache: Path | str | None = None,
) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    addr_path = address_index_checkpoint_path(index_dir, train_t)
    if addr_path.exists() and not force_refresh:
        logger.info("Stage A.5b cached: %s", addr_path)
        return addr_path

    print(f"\n=== Stage A.5b: address index pretrain T={train_t} ===")
    print(f"  dense teacher: {dense_ckpt}")
    meta = pretrain_addresses_on_dense_checkpoint(
        config,
        dense_ckpt,
        train_t,
        addr_path,
        device,
        dry_run=dry_run,
        train_cache=Path(train_cache) if train_cache else None,
        holdout_cache=Path(holdout_cache) if holdout_cache else None,
        force_refresh_cache=force_refresh,
    )
    print(f"  saved: {addr_path}  steps={meta.get('address_steps')}  skipped={meta.get('skipped')}")
    return addr_path


def _load_address_index_into_model(
    model: torch.nn.Module,
    var_config: dict,
    address_ckpt: Path,
    device: torch.device,
) -> None:
    book = ensure_address_book_on_model(model, var_config, device)
    load_checkpoint(address_ckpt, book, device=device, strict=False)
    attach_address_book_to_model(model, book)


def run_key_vector_k32(
    config: dict,
    *,
    train_t: int,
    dense_ckpt: Path,
    device: torch.device,
    log,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Stage B for key_vector (no index pretrain):
      Load dense trunk Q/K/V into KeyVectorSparseAttention layers; sparse top-k retrieval
      uses head-mean Q/K dot products (no RouterMLP, no address projections).
    """
    run_config = config
    k = top_k
    if k is None:
        k = int(config.get("key_vector", {}).get("top_k") or config.get("router", {}).get("top_k", 32))
    if int(config.get("router", {}).get("top_k", k)) != k:
        run_config = deepcopy(config)
        run_config.setdefault("router", {})["top_k"] = k

    bench_cfg, holdout_mid, holdout_full, holdout_meta = _arena_holdout_splits(run_config, train_t)

    sparse_steps = int(
        run_config.get("key_vector", {}).get("sparse_finetune_steps")
        or run_config.get("transformer", {}).get("sparse_finetune_steps")
        or run_config.get("transformer", {}).get("max_steps", 0)
    )
    sparse_lr = float(
        run_config.get("key_vector", {}).get("sparse_finetune_lr")
        or run_config.get("transformer", {}).get("lr", 3e-4)
    )

    print(f"\n=== key_vector_k32 Stage B: sparse finetune T={train_t} ===")
    print(f"  dense_checkpoint={dense_ckpt}")
    print(
        f"  holdout_mid={holdout_meta['holdout_mid_samples']}  "
        f"holdout_official={holdout_meta['holdout_full_samples']}"
    )
    print(
        f"  steps={sparse_steps} @ lr={sparse_lr} "
        f"(full trunk; top-k={k} on head-mean Q/K, no router/address index)"
    )

    model, var_config = _build_variant_model(
        run_config,
        "key_vector_k32",
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    apply_routing_variant_settings(model, var_config, "key_vector", train_t)

    routing_info = getattr(model, "_exp7_routing_info", {})
    audit = _verify_staged_training_protocol(
        "key_vector_k32",
        "finetune_from_dense",
        routing_info,
        model,
        two_stage=True,
    )
    print(f"  audit: {audit}")

    finetune_config = deepcopy(var_config)
    finetune_config.setdefault("transformer", {})["lr"] = sparse_lr

    train_info = _train_on_benchmark(
        model,
        finetune_config,
        bench_cfg,
        holdout_mid,
        device,
        train_t,
        log,
        max_steps=sparse_steps,
        training_stage="finetune_from_dense",
    )

    eval_official = _resolve_official_eval(
        run_config, model, bench_cfg, holdout_full, device, train_info
    )
    return {
        "variant": "key_vector_k32",
        "top_k": k,
        "holdout": holdout_meta,
        "training_audit": audit,
        "train_info": train_info,
        "eval": eval_official,
        "eval_official": eval_official,
        "routing_info": routing_info,
        "schedule": {
            "sparse_finetune_steps": sparse_steps,
            "sparse_finetune_lr": sparse_lr,
            "top_k": k,
            "index_pretrain": None,
        },
    }


def run_attention_baseline(
    config: dict,
    variant: str,
    *,
    train_t: int,
    dense_ckpt: Path | None,
    device: torch.device,
    log,
    save_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """
    Stage B for industry baselines (linear, local window):
      Load dense trunk → swap attention mechanism → LM fine-tune on NIAH.
      Uses production kernels: FLA chunk linear attn; Flex sliding-window local.
    """
    if variant not in BASELINE_ATTENTION_VARIANTS:
        raise ValueError(f"Not a baseline attention variant: {variant}")

    bench_cfg, holdout_mid, holdout_full, holdout_meta = _arena_holdout_splits(config, train_t)
    finetune_steps = int(
        config.get("transformer", {}).get("sparse_finetune_steps")
        or config.get("transformer", {}).get("max_steps", 0)
    )
    finetune_lr = float(config.get("transformer", {}).get("lr", 3e-4))

    backend = production_manifest_for_variants([variant]).get(variant, {})
    expected_cls, expected_window = _BASELINE_EXPECTED_ATTN[variant]

    from_scratch = dense_ckpt is None
    stage_label = "dense_pretrain" if from_scratch else "finetune_from_dense"
    print(f"\n=== {variant} {'Stage A' if from_scratch else 'Stage B'}: "
          f"{'train' if from_scratch else 'fine-tune'} T={train_t} ===")
    if from_scratch:
        print("  dense_checkpoint=None (train from scratch)")
    else:
        print(f"  dense_checkpoint={dense_ckpt}")
    print(f"  kernel={backend.get('kernel')}  package={backend.get('package', '')}")
    if expected_window is not None:
        print(f"  local_window={expected_window}")
    print(
        f"  holdout_mid={holdout_meta['holdout_mid_samples']}  "
        f"holdout_official={holdout_meta['holdout_full_samples']}"
    )
    print(f"  steps={finetune_steps} @ lr={finetune_lr} (full trunk)")

    model, var_config = _build_variant_model(
        config,
        variant,
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    attn_type = getattr(model, "attention_type", variant)
    routing_info = apply_routing_variant_settings(model, var_config, attn_type, train_t)
    routing_info.update(getattr(model, "_exp7_routing_info", {}))
    model._exp7_routing_info = routing_info  # type: ignore[attr-defined]

    n_expected = sum(1 for b in model.blocks if type(b.attn).__name__ == expected_cls)
    if n_expected != model.n_layers:
        raise RuntimeError(
            f"{variant}: expected {expected_cls} at all {model.n_layers} layers, got {n_expected}"
        )
    if expected_window is not None:
        window = int(getattr(model.blocks[0].attn, "window_size", 0))
        if window != expected_window:
            raise RuntimeError(f"{variant}: expected window={expected_window}, got {window}")

    audit = _verify_staged_training_protocol(
        variant,
        stage_label,
        routing_info,
        model,
        two_stage=True,
    )
    print(f"  audit: {audit}")

    finetune_config = deepcopy(var_config)
    finetune_config.setdefault("transformer", {})["lr"] = finetune_lr

    train_info = _train_on_benchmark(
        model,
        finetune_config,
        bench_cfg,
        holdout_mid,
        device,
        train_t,
        log,
        max_steps=finetune_steps,
        training_stage=stage_label,
    )

    eval_official = _resolve_official_eval(
        config, model, bench_cfg, holdout_full, device, train_info
    )
    saved_checkpoint: str | None = None
    if save_checkpoint_path is not None:
        from routing_attention.utils.checkpoint import save_checkpoint

        save_checkpoint_path = Path(save_checkpoint_path)
        save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            save_checkpoint_path,
            model,
            step=int(train_info.get("trained_steps", 0)),
            extra={
                "variant": variant,
                "attention_type": getattr(model, "attention_type", variant),
                "train_context_length": train_t,
                "training_stage": stage_label,
            },
        )
        saved_checkpoint = str(save_checkpoint_path)
        print(f"  saved variant checkpoint: {saved_checkpoint}")

    return {
        "variant": variant,
        "holdout": holdout_meta,
        "training_audit": audit,
        "train_info": train_info,
        "eval": eval_official,
        "eval_official": eval_official,
        "routing_info": routing_info,
        "production_backend": backend,
        "saved_variant_checkpoint": saved_checkpoint,
        "schedule": {
            "finetune_steps": finetune_steps,
            "finetune_lr": finetune_lr,
            "from_scratch": from_scratch,
        },
    }


def run_learned_address_k32(
    config: dict,
    *,
    train_t: int,
    dense_ckpt: Path,
    address_idx: Path,
    device: torch.device,
    log,
) -> dict[str, Any]:
    """
    Stage B for learned_address_k32 (after Stage A.5b address index pretrain):
      Load dense trunk + pretrained address projections; joint sparse top-k LM fine-tune
      (addresses + trunk trainable; meat Q/K/V copied from dense init).
    """
    bench_cfg, holdout_mid, holdout_full, holdout_meta = _arena_holdout_splits(config, train_t)

    joint_steps = int(
        config.get("learned_address", {}).get("joint_finetune_steps")
        or config.get("transformer", {}).get("sparse_finetune_steps", 5000)
    )
    joint_lr = float(
        config.get("learned_address", {}).get("joint_finetune_lr")
        or config.get("transformer", {}).get("lr", 3e-4)
    )

    print(f"\n=== learned_address_k32 Stage B: joint sparse finetune T={train_t} ===")
    print(f"  address_index={address_idx}")
    print(
        f"  holdout_mid={holdout_meta['holdout_mid_samples']}  "
        f"holdout_official={holdout_meta['holdout_full_samples']}"
    )
    print(f"  steps={joint_steps} @ lr={joint_lr} (trunk + address projections)")

    model, var_config = _build_variant_model(
        config,
        "learned_address_k32",
        device,
        train_t,
        dense_checkpoint=dense_ckpt,
    )
    _load_address_index_into_model(model, var_config, address_idx, device)
    var_config.setdefault("routing_attention", {})["freeze_addresses"] = False
    apply_routing_variant_settings(model, var_config, "learned_address", train_t)

    routing_info = getattr(model, "_exp7_routing_info", {})
    routing_info["address_index_checkpoint"] = str(address_idx)
    audit = _verify_staged_training_protocol(
        "learned_address_k32",
        "finetune_from_dense",
        routing_info,
        model,
        two_stage=True,
    )
    print(f"  audit: {audit}")

    joint_config = deepcopy(var_config)
    joint_config.setdefault("transformer", {})["lr"] = joint_lr

    train_joint = _train_on_benchmark(
        model,
        joint_config,
        bench_cfg,
        holdout_mid,
        device,
        train_t,
        log,
        max_steps=joint_steps,
        training_stage="finetune_from_dense",
    )

    eval_official = _resolve_official_eval(
        config, model, bench_cfg, holdout_full, device, train_joint
    )
    return {
        "variant": "learned_address_k32",
        "holdout": holdout_meta,
        "training_audit": audit,
        "train_joint": train_joint,
        "train_info": train_joint,
        "eval": eval_official,
        "eval_official": eval_official,
        "routing_info": routing_info,
        "address_index_checkpoint": str(address_idx),
        "schedule": {
            "joint_finetune_steps": joint_steps,
            "joint_finetune_lr": joint_lr,
        },
    }


def init_arena_runtime(config: dict) -> torch.device:
    training = config.get("training", {})
    deterministic = bool(training.get("cudnn_deterministic", False))
    device = init_experiment_runtime(config)
    set_seed(config.get("seed", 45), deterministic=deterministic)
    return device
