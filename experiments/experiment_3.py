"""
Experiment 3: Compare routing loss functions.

Trains separate routers with InfoNCE, MSE, and KL losses on the same attention cache.
Reuses transformer from Experiment 1; does NOT require Experiment 1 router.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm

from experiments.common import (
    announce_step,
    build_router,
    get_recall_eval_limits,
    init_experiment_runtime,
    load_experiment_config,
    load_transformer_from_reuse,
    resolve_attention_cache,
    resolve_eval_every,
    save_stats,
)
from routing_attention.evaluation.plots import plot_comparison_bar, save_all_plots
from routing_attention.evaluation.recall import compute_recall_at_k, compute_recall_from_router
from routing_attention.training.trainer import RouterTrainer
from routing_attention.utils.experiment import ExperimentRunner
from routing_attention.utils.logging import MetricsLogger, setup_logging


LOSS_VARIANTS = ("infonce", "infonce_sampled", "mse", "kl")


def run(
    variant: str | None = None,
    dry_run: bool = False,
    config_override: dict | None = None,
    compare_all_losses: bool = True,
) -> dict[str, Any]:
    config = load_experiment_config(3, variant=variant, config_override=config_override)
    device = init_experiment_runtime(config)

    runner = ExperimentRunner(
        experiment_name=config["experiment"]["name"],
        config=config,
        dry_run=dry_run,
    )
    config = runner.config
    logger = setup_logging(runner.run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)
    logger.info("Starting Experiment 3: %s", config["experiment"]["description"])

    announce_step("1 — Load transformer and attention cache", logger)
    model, _ = load_transformer_from_reuse(config, device)
    model.eval()

    cache_path = resolve_attention_cache(config, runner)
    if cache_path is None:
        raise FileNotFoundError("Run Experiment 1 first to generate attention cache.")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Use last-layer cache for single-router loss comparison
    if "layers" in cache:
        layer_idx = config["data_collection"].get("layer_idx", -1)
        if layer_idx < 0:
            layer_idx = config["model"]["n_layers"] + layer_idx
        layer_data = cache["layers"][layer_idx]
        single_cache_path = runner.data_cache_dir / f"loss_compare_layer_{layer_idx}.pt"
        torch.save(layer_data, single_cache_path)
        cache_path = single_cache_path
        hidden = layer_data["hidden_states"]
        attention = layer_data["attention"]
    else:
        hidden = cache["hidden_states"]
        attention = cache["attention"]
    top_k = config["router"]["top_k"]

    losses_to_run = list(LOSS_VARIANTS) if compare_all_losses else [config["router"]["loss_type"]]
    if dry_run:
        losses_to_run = losses_to_run[:2]
    max_eval_samples, max_eval_tokens = get_recall_eval_limits(config, dry_run=dry_run)
    comparison = {}

    announce_step(
        "2 — Train routers per loss function",
        logger,
        f"{len(losses_to_run)} losses × {config['router']['max_steps']} steps",
    )
    for loss_type in tqdm(losses_to_run, desc="Loss comparison", unit="loss"):
        announce_step(f"Train router — {loss_type}", logger, f"{config['router']['max_steps']} steps")
        logger.info("Training router with %s loss", loss_type)
        loss_cfg = {**config, "router": {**config["router"], "loss_type": loss_type, "mode": "single"}}
        router = build_router(loss_cfg).to(device)
        optimizer = optim.AdamW(router.parameters(), lr=config["router"]["lr"])

        loss_config = {**config["router"], "loss_type": loss_type}
        runner_sub = runner  # same run dir, different checkpoints

        router_eval_every = resolve_eval_every(config, "router")
        eval_max_samples = config.get("validation", {}).get("eval_max_samples", 8)

        def router_eval_fn(r, h, attn):
            r.eval()
            hh, aa = h, attn
            if eval_max_samples > 0 and hh.shape[0] > eval_max_samples:
                idx = torch.randperm(hh.shape[0], device=hh.device)[:eval_max_samples]
                hh, aa = hh[idx], aa[idx]
            with torch.no_grad():
                routing = r(hh)
                return compute_recall_at_k(routing, aa, k=top_k)

        trainer = RouterTrainer(
            router=router,
            optimizer=optimizer,
            device=device,
            runner=runner_sub,
            metrics_logger=metrics_logger,
            logger=logger,
            loss_type=loss_type,
            top_k=top_k,
            temperature=config["router"]["temperature"],
            max_steps=config["router"]["max_steps"],
            eval_every=router_eval_every,
            save_every=config["router"]["save_every"],
            eval_fn=router_eval_fn if router_eval_every > 0 else None,
            eval_max_samples=eval_max_samples,
            use_amp=config.get("training", {}).get("use_amp", True),
        )
        train_result = trainer.train_on_cache(cache_path)

        router.eval()
        recall = compute_recall_from_router(
            hidden,
            attention,
            router,
            k=top_k,
            max_samples=max_eval_samples,
            max_tokens=max_eval_tokens,
        )

        comparison[loss_type] = {
            "training": train_result,
            "recall": recall,
        }
        save_stats(runner.stats_dir, f"loss_{loss_type}", comparison[loss_type])

        # Save per-loss router checkpoint
        from routing_attention.utils.checkpoint import save_checkpoint
        save_checkpoint(
            runner.checkpoint_dir / f"router_{loss_type}.pt",
            router,
            optimizer,
            metrics=recall,
            extra={"loss_type": loss_type},
        )

    # Determine best loss
    recall_key = f"recall@{min(top_k, hidden.shape[1])}"
    best_loss = max(comparison.keys(), key=lambda k: comparison[k]["recall"].get(recall_key, 0))
    save_stats(runner.stats_dir, "loss_comparison", comparison)
    save_stats(runner.stats_dir, "verdict", {"best_loss_type": best_loss})

    recall_bar = {k: v["recall"].get(recall_key, 0) for k, v in comparison.items()}
    plot_comparison_bar(
        recall_bar,
        runner.plots_dir / "loss_comparison_recall.png",
        title="Recall@K by Loss Function",
        ylabel=recall_key,
    )

    plot_paths = save_all_plots(runner.plots_dir, {"metrics_history": metrics_logger._history})
    plot_paths.append(str(runner.plots_dir / "loss_comparison_recall.png"))
    metrics_logger.close()

    summary = {
        "experiment": "Experiment_3",
        "run_dir": str(runner.run_dir),
        "variant": variant or "default",
        "comparison": {k: v["recall"] for k, v in comparison.items()},
        "best_loss_type": best_loss,
        "plot_paths": plot_paths,
    }
    runner.finalize(summary)
    logger.info("Experiment 3 complete. Best loss: %s", best_loss)
    return summary


if __name__ == "__main__":
    run(dry_run=True)
