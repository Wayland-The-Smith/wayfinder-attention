"""
Experiment 2: Long-range vs local attention recovery.

Fixes mid_layer variant: re-collects cache at target layer and retrains router if needed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim

from experiments.common import (
    announce_step,
    build_router,
    get_recall_eval_limits,
    get_train_dataloader,
    init_experiment_runtime,
    load_experiment_config,
    load_router_from_reuse,
    load_transformer_from_reuse,
    make_router_eval_fn,
    resolve_attention_cache,
    resolve_eval_every,
    save_stats,
)
from routing_attention.evaluation.plots import save_all_plots
from routing_attention.evaluation.recall import compute_recall_by_distance_from_router
from routing_attention.models.router import PerLayerRouter
from routing_attention.training.trainer import RouterTrainer, collect_attention_dataset
from routing_attention.utils.experiment import ExperimentRunner
from routing_attention.utils.logging import MetricsLogger, setup_logging


def run(
    variant: str | None = None,
    dry_run: bool = False,
    config_override: dict | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(2, variant=variant, config_override=config_override)
    device = init_experiment_runtime(config)

    runner = ExperimentRunner(
        experiment_name=config["experiment"]["name"],
        config=config,
        dry_run=dry_run,
    )
    config = runner.config
    logger = setup_logging(runner.run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)
    logger.info("Starting Experiment 2: %s", config["experiment"]["description"])

    layer_idx = config["data_collection"]["layer_idx"]
    if layer_idx < 0:
        layer_idx = config["model"]["n_layers"] + layer_idx

    announce_step("1 — Load frozen transformer", logger)
    model, _ = load_transformer_from_reuse(config, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    announce_step("2 — Resolve attention cache", logger, f"target layer {layer_idx}")
    cache_path = runner.data_cache_dir / f"layer_{layer_idx}_cache.pt"
    if not cache_path.exists():
        logger.info("Collecting attention cache for layer %d", layer_idx)
        all_layers_cache = resolve_attention_cache(config, runner)
        if all_layers_cache and all_layers_cache.name == "attention_cache_all_layers.pt":
            full = torch.load(all_layers_cache, map_location="cpu", weights_only=False)
            if "layers" in full and layer_idx in full["layers"]:
                torch.save(full["layers"][layer_idx], cache_path)
                logger.info("Extracted layer %d from all_layers cache", layer_idx)
            else:
                collect_attention_dataset(
                    model, get_train_dataloader(config, for_collection=True), device,
                    layer_idx=layer_idx,
                    max_batches=config["data_collection"]["max_batches"],
                    output_path=cache_path,
                    per_head=config["data_collection"].get("per_head", False),
                )
        else:
            collect_attention_dataset(
                model, get_train_dataloader(config, for_collection=True), device,
                layer_idx=layer_idx,
                max_batches=config["data_collection"]["max_batches"],
                output_path=cache_path,
                per_head=config["data_collection"].get("per_head", False),
            )

    announce_step("3 — Load or train router", logger, f"layer {layer_idx}")
    router_cfg = config["router"]
    retrain = config.get("experiment", {}).get("retrain_router", True)
    router, router_meta = load_router_from_reuse(config, device)

    # If per-layer router from Exp 1, use matching layer without retraining
    if isinstance(router, PerLayerRouter) and router_meta is not None:
        retrain = False
        eval_router = router.get_router(layer_idx)
        logger.info("Using per-layer router weights for layer %d", layer_idx)
    elif retrain and router_meta is None:
        logger.info("Training router for layer %d", layer_idx)
        if isinstance(router, PerLayerRouter):
            train_router = router.get_router(layer_idx)
        else:
            train_router = build_router({**config, "router": {**router_cfg, "mode": "single"}})

        optimizer = optim.AdamW(train_router.parameters(), lr=router_cfg["lr"])
        orig_dir = runner.checkpoint_dir / "router" / f"layer_{layer_idx}"
        orig_dir.mkdir(parents=True, exist_ok=True)
        runner.checkpoint_dir = orig_dir

        router_eval_every = resolve_eval_every(config, "router")
        eval_max_samples = config.get("validation", {}).get("eval_max_samples", 8)
        eval_fn = (
            make_router_eval_fn(router_cfg["top_k"], layer_idx, eval_max_samples=eval_max_samples)
            if router_eval_every > 0 else None
        )
        trainer = RouterTrainer(
            router=train_router,
            optimizer=optimizer,
            device=device,
            runner=runner,
            metrics_logger=metrics_logger,
            logger=logger,
            loss_type=router_cfg["loss_type"],
            top_k=router_cfg["top_k"],
            temperature=router_cfg["temperature"],
            max_steps=router_cfg["max_steps"],
            eval_every=router_eval_every,
            save_every=router_cfg["save_every"],
            eval_fn=eval_fn,
            batch_size=router_cfg.get("batch_size", 4),
            layer_idx=layer_idx,
            eval_max_samples=eval_max_samples,
            use_amp=config.get("training", {}).get("use_amp", True),
        )
        trainer.train_on_cache(cache_path)
        eval_router = train_router
    elif not retrain:
        if isinstance(router, PerLayerRouter):
            eval_router = router.get_router(layer_idx)
        else:
            eval_router = router
    else:
        eval_router = router.get_router(layer_idx) if isinstance(router, PerLayerRouter) else router

    max_eval_samples, max_eval_tokens = get_recall_eval_limits(config, dry_run=dry_run)
    recall_k = config["evaluation"].get("recall_k", router_cfg["top_k"])
    announce_step(
        "4 — Recall by distance evaluation",
        logger,
        f"k={recall_k}, samples={max_eval_samples or 'all'}, tokens={max_eval_tokens or 'all'}",
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    hidden = cache["hidden_states"]
    attention = cache["attention"]

    eval_router.eval()
    distance_results = compute_recall_by_distance_from_router(
        hidden,
        attention,
        eval_router,
        k=recall_k,
        max_samples=max_eval_samples,
        max_tokens=max_eval_tokens,
    )

    save_stats(runner.stats_dir, "recall_by_distance", distance_results)
    save_stats(runner.stats_dir, "overall_recall", distance_results["overall"])
    save_stats(runner.stats_dir, "layer_idx", {"layer_idx": layer_idx})

    per_bin = distance_results["per_bin"]
    long_range_bins = [b for b in per_bin if b["distance_min"] >= 64]
    long_range_recall = (
        sum(b["hits"] for b in long_range_bins) / max(sum(b["total"] for b in long_range_bins), 1)
        if long_range_bins else 0.0
    )

    verdict = "strong_long_range" if long_range_recall >= 0.5 else (
        "local_only" if long_range_recall < 0.2 else "mixed"
    )
    save_stats(runner.stats_dir, "verdict", {"long_range_recall": long_range_recall, "assessment": verdict, "layer_idx": layer_idx})

    plot_paths = save_all_plots(
        runner.plots_dir,
        {"recall_by_distance": distance_results, "metrics_history": metrics_logger._history},
    )
    metrics_logger.log_scalar("eval/long_range_recall", long_range_recall, 0)
    metrics_logger.close()

    summary = {
        "experiment": "Experiment_2",
        "run_dir": str(runner.run_dir),
        "variant": variant or "default",
        "layer_idx": layer_idx,
        "overall_recall": distance_results["overall"],
        "long_range_recall": long_range_recall,
        "verdict": verdict,
        "plot_paths": plot_paths,
    }
    runner.finalize(summary)
    logger.info("Experiment 2 complete (layer %d). Long-range recall: %.3f (%s)", layer_idx, long_range_recall, verdict)
    return summary


if __name__ == "__main__":
    run(dry_run=True)
