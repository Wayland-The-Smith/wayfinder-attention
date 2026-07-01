"""Stage A.5 — train routing index (router MLP) on NIAH dense teacher at fixed T."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from experiments.common import (
    build_router,
    build_transformer,
    collect_router_attention_caches,
)
from experiments.experiment_4 import _expand_position_embeddings, _load_state_dict_tolerant
from routing_attention.training.trainer import train_per_layer_routers
from routing_attention.utils.checkpoint import save_checkpoint
from routing_attention.utils.experiment import ExperimentRunner
from routing_attention.utils.logging import MetricsLogger, setup_logging

logger = logging.getLogger(__name__)


def router_index_checkpoint_path(checkpoint_dir: Path, train_t: int) -> Path:
    return checkpoint_dir / f"T{train_t}_routing_index.pt"


def address_index_checkpoint_path(checkpoint_dir: Path, train_t: int) -> Path:
    return checkpoint_dir / f"T{train_t}_address_index.pt"


def _index_pretrain_config(config: dict, train_t: int, dry_run: bool) -> dict:
    cfg = deepcopy(config)
    cfg.setdefault("data", {})["train_context_length"] = train_t
    cfg["data"]["seq_len"] = train_t
    cfg["model"]["max_seq_len"] = train_t
    idx_cfg = cfg.setdefault("index_pretrain", {})
    coll = cfg.setdefault("data_collection", {})
    if dry_run:
        coll["train_max_batches"] = int(idx_cfg.get("dry_run_cache_train_batches", 4))
        coll["holdout_max_batches"] = int(idx_cfg.get("dry_run_cache_holdout_batches", 2))
        cfg.setdefault("router", {})["max_steps"] = int(idx_cfg.get("dry_run_router_steps", 20))
        cfg["router"]["eval_every"] = 0
    else:
        coll["train_max_batches"] = int(
            idx_cfg.get("cache_train_batches", coll.get("train_max_batches", 32))
        )
        coll["holdout_max_batches"] = int(
            idx_cfg.get("cache_holdout_batches", coll.get("holdout_max_batches", 8))
        )
        router_steps = idx_cfg.get("router_max_steps")
        if router_steps is not None:
            cfg.setdefault("router", {})["max_steps"] = int(router_steps)
    return cfg


def pretrain_router_on_dense_checkpoint(
    config: dict,
    dense_checkpoint: Path,
    train_t: int,
    save_path: Path,
    device: torch.device,
    *,
    dry_run: bool = False,
    force_refresh_cache: bool = False,
) -> dict[str, Any]:
    """
    Collect dense attention neighborhoods on NIAH at ``train_t``, train per-layer router.

    Returns metadata dict; saves router weights to ``save_path``.
    """
    save_path = Path(save_path)
    if save_path.exists() and not force_refresh_cache:
        return {"path": str(save_path), "skipped": True, "train_context_length": train_t}

    cfg = _index_pretrain_config(config, train_t, dry_run)
    run_dir = save_path.parent / f"_index_cache_T{train_t}"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(
        experiment_name=cfg["experiment"]["name"],
        config=cfg,
        dry_run=dry_run,
        run_dir=run_dir,
    )
    run_logger = setup_logging(run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)

    run_logger.info("Stage A.5: router index pretrain at T=%d from %s", train_t, dense_checkpoint)

    model = build_transformer(cfg, attention_type="dense_flash").to(device)
    _expand_position_embeddings(model, train_t)
    payload = _load_state_dict_tolerant(model, dense_checkpoint, device)
    if payload is None:
        raise FileNotFoundError(f"Dense checkpoint missing for index pretrain: {dense_checkpoint}")
    model.eval()

    train_cache, holdout_cache = collect_router_attention_caches(
        model,
        cfg,
        runner,
        device,
        logger=run_logger,
        all_layers=cfg.get("data_collection", {}).get("all_layers", True),
        force_refresh=force_refresh_cache,
    )
    run_logger.info("Attention caches: train=%s holdout=%s", train_cache, holdout_cache)

    router = build_router(cfg).to(device)
    router_opt = optim.AdamW(router.parameters(), lr=cfg["router"]["lr"])
    runner.checkpoint_dir = runner.checkpoint_dir / "router"
    runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_per_layer_routers(
        router=router,
        cache_path=train_cache,
        optimizer=router_opt,
        device=device,
        runner=runner,
        metrics_logger=metrics_logger,
        logger=run_logger,
        config=cfg,
        eval_fn=None,
        holdout_cache_path=holdout_cache,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_path,
        router,
        extra={
            "train_context_length": train_t,
            "dense_checkpoint": str(dense_checkpoint),
            "training_stage": "index_pretrain",
            "task": "long_context_niah",
        },
    )
    run_logger.info("Saved task router index: %s", save_path)

    del model, router
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "path": str(save_path),
        "train_context_length": train_t,
        "dense_checkpoint": str(dense_checkpoint),
        "train_cache": str(train_cache),
        "holdout_cache": str(holdout_cache),
        "router_steps": cfg["router"]["max_steps"],
        "skipped": False,
    }


def pretrain_addresses_on_dense_checkpoint(
    config: dict,
    dense_checkpoint: Path,
    train_t: int,
    save_path: Path,
    device: torch.device,
    *,
    dry_run: bool = False,
    train_cache: Path | None = None,
    holdout_cache: Path | None = None,
    force_refresh_cache: bool = False,
) -> dict[str, Any]:
    """
    Stage A.5b — train per-layer address projections (InfoNCE) on dense teacher caches.

    Reuses attention caches from router index pretrain when available.
    """
    from experiments.common import build_address_book
    from routing_attention.training.trainer import train_per_layer_addresses

    save_path = Path(save_path)
    if save_path.exists() and not force_refresh_cache:
        return {"path": str(save_path), "skipped": True, "train_context_length": train_t}

    cfg = _index_pretrain_config(config, train_t, dry_run)
    run_dir = save_path.parent / f"_address_cache_T{train_t}"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(
        experiment_name=cfg["experiment"]["name"],
        config=cfg,
        dry_run=dry_run,
        run_dir=run_dir,
    )
    run_logger = setup_logging(run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)

    if train_cache is None or holdout_cache is None or force_refresh_cache:
        model = build_transformer(cfg, attention_type="dense_flash").to(device)
        _expand_position_embeddings(model, train_t)
        payload = _load_state_dict_tolerant(model, dense_checkpoint, device)
        if payload is None:
            raise FileNotFoundError(f"Dense checkpoint missing for address index pretrain: {dense_checkpoint}")
        model.eval()
        train_cache, holdout_cache = collect_router_attention_caches(
            model,
            cfg,
            runner,
            device,
            logger=run_logger,
            all_layers=cfg.get("data_collection", {}).get("all_layers", True),
            force_refresh=force_refresh_cache,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_logger.info(
        "Stage A.5b: address index pretrain at T=%d (train_cache=%s)",
        train_t,
        train_cache,
    )

    address_book = build_address_book(cfg).to(device)
    idx_cfg = cfg.setdefault("index_pretrain", {})
    addr_steps = int(
        idx_cfg.get("address_index_steps") or idx_cfg.get("router_max_steps", 2000)
    )
    cfg.setdefault("router", {})["max_steps"] = addr_steps
    addr_opt = optim.AdamW(address_book.parameters(), lr=cfg["router"]["lr"])
    runner.checkpoint_dir = runner.checkpoint_dir / "addresses"
    runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_per_layer_addresses(
        address_book=address_book,
        cache_path=train_cache,
        optimizer=addr_opt,
        device=device,
        runner=runner,
        metrics_logger=metrics_logger,
        logger=run_logger,
        config=cfg,
        eval_fn=None,
        holdout_cache_path=holdout_cache,
    )

    recall_k = int(cfg.get("router", {}).get("top_k", 32))
    n_layers = int(cfg.get("model", {}).get("n_layers", 4))
    recall_metrics: dict[str, Any] = {}
    try:
        from routing_attention.evaluation.recall import evaluate_learned_address_recall_from_cache

        max_tokens = 128 if dry_run else 0
        recall_metrics = evaluate_learned_address_recall_from_cache(
            address_book,
            holdout_cache,
            device,
            recall_k=recall_k,
            n_layers=n_layers,
            max_eval_tokens=max_tokens,
            show_progress=not dry_run,
        )
        run_logger.info(
            "Phase B holdout Recall@%d mean=%.4f",
            recall_k,
            float(recall_metrics.get("mean_recall", 0.0)),
        )
    except Exception as exc:
        run_logger.warning("Address recall eval failed: %s", exc)
        recall_metrics = {"error": str(exc)}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_path,
        address_book,
        extra={
            "train_context_length": train_t,
            "dense_checkpoint": str(dense_checkpoint),
            "training_stage": "address_index_pretrain",
            "task": "long_context_niah",
        },
    )
    run_logger.info("Saved task address index: %s", save_path)

    del address_book
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "path": str(save_path),
        "train_context_length": train_t,
        "dense_checkpoint": str(dense_checkpoint),
        "train_cache": str(train_cache),
        "holdout_cache": str(holdout_cache),
        "address_steps": addr_steps,
        "recall_k": recall_k,
        "holdout_recall": recall_metrics,
        "skipped": False,
    }
