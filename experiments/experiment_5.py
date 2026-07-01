"""
Experiment 5: MNIST routing quality and retrieval throughput.

Measures recall, pixel-LM loss, and attention throughput on MNIST (784 tokens)
across retrieval backend variants. Reuses models from Experiments 1 and 4.
"""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from experiments.common import (
    announce_step,
    get_benchmark_eval_config,
    get_collection_dataloader,
    get_eval_dataloader,
    get_recall_eval_limits,
    init_experiment_runtime,
    load_experiment_config,
    load_router_from_reuse,
    load_transformer_from_reuse,
    resolve_routing_model_checkpoint,
    save_stats,
)
from routing_attention.evaluation.benchmarking import benchmark_attention
from routing_attention.evaluation.metrics import evaluate_lm
from routing_attention.evaluation.plots import plot_scale_curves, save_all_plots
from routing_attention.evaluation.recall import compute_recall_from_router, evaluate_per_layer_recall
from routing_attention.models.transformer import TransformerLM
from routing_attention.training.trainer import collect_attention_dataset
from routing_attention.utils.checkpoint import load_checkpoint
from routing_attention.utils.config import apply_variant
from routing_attention.utils.experiment import ExperimentRunner
from routing_attention.utils.logging import MetricsLogger, setup_logging


RETRIEVAL_VARIANTS = ("default", "brute_force", "faiss_hnsw")


def run(
    variant: str | None = None,
    dry_run: bool = False,
    config_override: dict | None = None,
    compare_all: bool = True,
) -> dict[str, Any]:
    config = load_experiment_config(5, variant=variant, config_override=config_override)
    device = init_experiment_runtime(config)

    runner = ExperimentRunner(
        experiment_name=config["experiment"]["name"],
        config=config,
        dry_run=dry_run,
    )
    config = runner.config
    logger = setup_logging(runner.run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)
    logger.info("Starting Experiment 5: %s", config["experiment"]["description"])

    variants_to_run = list(RETRIEVAL_VARIANTS) if compare_all and variant is None else [variant or "default"]
    if dry_run:
        variants_to_run = variants_to_run[:1]

    bench_cfg = get_benchmark_eval_config(config)
    backend_results: dict[str, Any] = {}
    seq_len = config["data"]["seq_len"]

    announce_step("1 — Load models from Experiments 1 and 4", logger)
    base_model, _ = load_transformer_from_reuse(config, device, attention_type="dense")
    router, _ = load_router_from_reuse(config, device)

    announce_step(
        "2 — MNIST eval across retrieval backends",
        logger,
        f"{len(variants_to_run)} variants, seq_len={seq_len}",
    )
    for var_name in tqdm(variants_to_run, desc="Retrieval backends", unit="variant"):
        var_config = apply_variant(config, var_name)
        announce_step(f"MNIST backend test — {var_name}", logger)
        logger.info("Evaluating retrieval method=%s on MNIST test split", var_config["retrieval"]["method"])

        eval_router, _ = load_router_from_reuse(var_config, device)
        model = TransformerLM(
            vocab_size=base_model.lm_head.out_features,
            d_model=var_config["model"]["d_model"],
            n_layers=var_config["model"]["n_layers"],
            n_heads=var_config["model"]["n_heads"],
            d_ff=var_config["model"]["d_ff"],
            max_seq_len=seq_len,
            dropout=var_config["model"]["dropout"],
            attention_type="routing",
            router=eval_router,
            routing_top_k=var_config["router"]["top_k"],
            retrieval_config=var_config.get("retrieval"),
        ).to(device)

        routing_ckpt = resolve_routing_model_checkpoint(var_config)
        if routing_ckpt is not None:
            load_checkpoint(routing_ckpt, model, device=device, strict=False)

        eval_dl = get_eval_dataloader(var_config)
        coll_dl = get_collection_dataloader(var_config, split="holdout")
        dense_model, _ = load_transformer_from_reuse(var_config, device, attention_type="dense")

        cache_path = runner.data_cache_dir / f"attention_cache_mnist_{var_name}.pt"
        all_layers_path = runner.data_cache_dir / f"attention_cache_mnist_{var_name}_all_layers.pt"
        recall_batches = var_config["evaluation"].get(
            "recall_max_batches",
            var_config["data_collection"]["holdout_max_batches"],
        )
        if not all_layers_path.exists():
            collect_attention_dataset(
                dense_model,
                coll_dl,
                device,
                max_batches=recall_batches,
                output_path=all_layers_path,
                all_layers=True,
                split="holdout",
            )

        cache = torch.load(all_layers_path, map_location="cpu", weights_only=False)
        from routing_attention.models.router import PerLayerRouter

        eval_router.eval()
        max_eval_samples, max_eval_tokens = get_recall_eval_limits(var_config, dry_run=dry_run)
        top_k = var_config["router"]["top_k"]
        recall_key = f"recall@{top_k}"
        if "layers" in cache and isinstance(eval_router, PerLayerRouter):
            recall = evaluate_per_layer_recall(
                router=eval_router,
                cache=cache,
                device=device,
                recall_k=top_k,
                n_layers=eval_router.n_layers,
                max_eval_samples=max_eval_samples,
                max_eval_tokens=max_eval_tokens,
                show_progress=False,
            )
            recall = {"per_layer": recall["per_layer"], "mean": recall["mean_recall"]}
        else:
            li = var_config["model"]["n_layers"] - 1
            ld = cache["layers"][li] if "layers" in cache else cache
            recall = compute_recall_from_router(
                ld["hidden_states"],
                ld["attention"],
                eval_router,
                layer_idx=li if isinstance(eval_router, PerLayerRouter) else None,
                k=top_k,
                max_samples=max_eval_samples,
                max_tokens=max_eval_tokens,
            )

        lm_metrics = evaluate_lm(model, eval_dl, device, max_batches=var_config["evaluation"]["max_batches"])
        bench = benchmark_attention(
            model,
            seq_len=seq_len,
            batch_size=var_config["data"]["batch_size"],
            device=device,
            num_runs=bench_cfg["num_runs"],
            num_warmup=bench_cfg["num_warmup"],
            retrieval_cfg=var_config.get("retrieval"),
        )

        backend_results[var_name] = {
            "retrieval_method": var_config["retrieval"]["method"],
            "seq_len": seq_len,
            "recall": recall,
            "lm": lm_metrics,
            "benchmark": bench,
        }
        save_stats(runner.stats_dir, f"mnist_{var_name}", backend_results[var_name])
        if isinstance(recall, dict) and "mean" in recall:
            metrics_logger.log_scalar(f"mnist_{var_name}/recall_mean", recall["mean"], 0)
        elif isinstance(recall, dict):
            metrics_logger.log_scalar(f"mnist_{var_name}/{recall_key}", recall.get(recall_key, 0), 0)
        metrics_logger.log_scalar(f"mnist_{var_name}/pixel_lm_loss", lm_metrics["loss"], 0)

    variant_names = list(backend_results.keys())
    backend_labels = [backend_results[v]["retrieval_method"] for v in variant_names]
    summary_curves = {
        "backends": backend_labels,
        "metrics": {
            "recall": [
                backend_results[v]["recall"].get("mean", backend_results[v]["recall"].get(recall_key, 0))
                if isinstance(backend_results[v]["recall"], dict) else backend_results[v]["recall"]
                for v in variant_names
            ],
            "pixel_lm_loss": [backend_results[v]["lm"]["loss"] for v in variant_names],
            "throughput": [backend_results[v]["benchmark"]["throughput_tokens_per_sec"] for v in variant_names],
            "latency_ms": [backend_results[v]["benchmark"]["latency_ms"] for v in variant_names],
        },
        "ylabel": "value",
    }
    save_stats(runner.stats_dir, "mnist_backend_curves", summary_curves)

    for metric_name, values in summary_curves["metrics"].items():
        plot_scale_curves(
            list(range(1, len(variant_names) + 1)),
            {metric_name: values},
            runner.plots_dir / f"mnist_{metric_name}.png",
            title=f"MNIST {metric_name} by retrieval backend",
            ylabel=metric_name,
        )

    best_var = max(
        variant_names,
        key=lambda v: backend_results[v]["recall"].get(
            "mean",
            backend_results[v]["recall"].get(recall_key, 0) if isinstance(backend_results[v]["recall"], dict) else 0,
        ),
    )
    best_recall_val = backend_results[best_var]["recall"]
    best_recall = (
        best_recall_val.get("mean", best_recall_val.get(recall_key, 0))
        if isinstance(best_recall_val, dict) else 0
    )
    criteria = config.get("success_criteria", {})
    verdict = "strong_mnist_routing" if best_recall >= criteria.get("recall_at_32", 0.50) else (
        "promising_mnist_routing" if best_recall >= 0.35 else "weak_mnist_routing"
    )
    save_stats(runner.stats_dir, "verdict", {"best_recall": best_recall, "best_variant": best_var, "assessment": verdict})

    plot_paths = save_all_plots(runner.plots_dir, {"mnist_backend_curves": summary_curves})
    metrics_logger.close()

    summary = {
        "experiment": "Experiment_5",
        "run_dir": str(runner.run_dir),
        "variant": variant or "compare_all",
        "backend_results": backend_results,
        "mnist_backend_curves": summary_curves,
        "verdict": verdict,
        "plot_paths": plot_paths,
    }
    runner.finalize(summary)
    logger.info("Experiment 5 complete. Best MNIST recall: %.3f (%s)", best_recall, verdict)
    return summary


if __name__ == "__main__":
    run(dry_run=True)
