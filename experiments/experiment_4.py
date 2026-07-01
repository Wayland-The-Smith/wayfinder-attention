"""
Experiment 4: Controlled head-to-head attention mechanism comparison.

Fair comparison rules:
  - All variants init from same dense transformer checkpoint
  - All variants receive identical fine-tune steps (fair_finetune: true)
  - Routing uses vector-search retrieval on learned routing vectors at EVERY layer
  - Router frozen by default during fine-tune (prevents geometry drift)
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm

from experiments.common import (
    announce_step,
    build_transformer,
    get_benchmark_eval_config,
    get_recall_eval_limits,
    get_eval_dataloader,
    get_train_dataloader,
    init_experiment_runtime,
    load_experiment_config,
    load_addresses_from_reuse,
    load_router_from_reuse,
    load_transformer_from_reuse,
    resolve_attention_cache,
    resolve_eval_every,
    save_stats,
    write_idea_manifest,
)
from routing_attention.evaluation.benchmarking import benchmark_attention
from routing_attention.evaluation.metrics import compare_attention_types, evaluate_lm
from routing_attention.evaluation.recall import (
    compute_recall_from_router,
    evaluate_per_layer_recall,
)
from routing_attention.models.router import MultiScaleRouter, PerLayerRouter
from routing_attention.models.transformer import TransformerLM
from routing_attention.retrieval.index import patch_model_retrievers, sync_retrieval_config_dict
from routing_attention.training.trainer import TransformerTrainer
from routing_attention.utils.checkpoint import load_checkpoint, save_checkpoint


def _load_state_dict_tolerant(model: TransformerLM, path: Path, device) -> dict | None:
    """Load matching-shape weights only (skips pos_emb when seq_len differs)."""
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    src = payload.get("model_state_dict", payload)
    dst = model.state_dict()
    filtered = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
    dst.update(filtered)
    model.load_state_dict(dst)
    return payload


def _benchmark_batch_size(base_batch: int, seq_len: int) -> int:
    """Reduce batch size at long sequence lengths to avoid OOM during scaling sweeps."""
    if seq_len >= 8192:
        return 1
    if seq_len >= 4096:
        return min(base_batch, 2)
    if seq_len >= 2048:
        return min(base_batch, 4)
    return base_batch


def _expand_position_embeddings(model: TransformerLM, min_seq_len: int) -> None:
    """Extend positional embeddings when benchmarking longer sequences."""
    old = model.pos_emb
    if old.num_embeddings >= min_seq_len:
        return
    d_model = old.embedding_dim
    new_emb = torch.nn.Embedding(min_seq_len, d_model, device=old.weight.device)
    with torch.no_grad():
        n_old = old.num_embeddings
        new_emb.weight[:n_old] = old.weight
        if min_seq_len > n_old:
            new_emb.weight[n_old:] = old.weight[-1]
    model.pos_emb = new_emb
from routing_attention.utils.experiment import ExperimentRunner
from routing_attention.utils.logging import MetricsLogger, setup_logging


# Each entry: variant name -> attention type
ATTENTION_VARIANTS = {
    "dense": "dense",
    "dense_flash": "dense_flash",
    "linear": "linear",
    "local_window256": "local",
    "routing": "routing",
    "routing_asymmetric": "routing",
    "routing_aux_loss": "routing",
    "local_window64": "local",
    "key_vector_k32": "key_vector",
    "learned_address_k32": "learned_address",
}


def run(
    variant: str | None = None,
    dry_run: bool = False,
    config_override: dict | None = None,
    compare_all: bool = True,
    variants: list[str] | None = None,
    skip_finetune: bool = False,
    use_best_checkpoint: bool = False,
    benchmark_seq_lens: list[int] | None = None,
    load_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(4, variant=variant, config_override=config_override)
    device = init_experiment_runtime(config)

    runner = ExperimentRunner(
        experiment_name=config["experiment"]["name"],
        config=config,
        dry_run=dry_run,
    )
    config = runner.config
    logger = setup_logging(runner.run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)
    logger.info("Starting Experiment 4: %s", config["experiment"]["description"])

    idea_manifest = config.get("idea_manifest")
    if idea_manifest:
        manifest_path = write_idea_manifest(runner.run_dir, idea_manifest)
        logger.info("Wrote idea manifest: %s", manifest_path)

    # Controlled comparison set — one routing variant, not duplicates
    if variants is not None:
        variants_to_run = list(variants)
    elif compare_all:
        variants_to_run = ["dense", "routing", "local_window64", "key_vector_k32", "learned_address_k32"]
        if dry_run:
            variants_to_run = variants_to_run[:2]
    else:
        variants_to_run = [variant or "routing"]

    max_eval_samples, max_eval_tokens = get_recall_eval_limits(config, dry_run=dry_run)
    bench_cfg = get_benchmark_eval_config(config)

    lm_results = {}
    benchmark_results = {}
    recall_results = {}
    training_info = {}

    announce_step("1 — Load shared transformer and teacher", logger)
    load_cfg = config
    if benchmark_seq_lens:
        load_cfg = deepcopy(config)
        load_cfg["model"]["max_seq_len"] = 784
    base_model, base_meta = load_transformer_from_reuse(load_cfg, device, attention_type="dense")
    if base_meta is None:
        logger.warning("No pretrained transformer — comparisons will not be controlled!")

    teacher_model, _ = load_transformer_from_reuse(load_cfg, device, attention_type="dense")
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    cache_path = resolve_attention_cache(config, runner)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path else None

    ra_base = config.get("routing_attention", {})
    fair_finetune = ra_base.get("fair_finetune", True)
    shared_max_steps = ra_base.get("max_steps", 10000)
    if skip_finetune or shared_max_steps <= 0:
        skip_finetune = True
        shared_max_steps = 0

    announce_step(
        "2 — Fair comparison across attention variants",
        logger,
        f"{len(variants_to_run)} variants × {shared_max_steps} steps each",
    )
    for var_name in tqdm(variants_to_run, desc="Attention variants", unit="variant"):
        announce_step(f"Variant — {var_name}", logger, f"{shared_max_steps} fine-tune steps")
        logger.info("=== Variant: %s ===", var_name)
        attn_type = ATTENTION_VARIANTS.get(var_name, "routing")
        var_config = _apply_variant_config(config, var_name)
        ra_cfg = var_config.get("routing_attention", {})

        # Build model — fresh from base weights each time
        if attn_type == "routing":
            router, _ = load_router_from_reuse(var_config, device)
            model = build_transformer(var_config, attention_type="routing", router=router).to(device)
        else:
            model = build_transformer(var_config, attention_type=attn_type).to(device)

        if attn_type == "learned_address":
            load_addresses_from_reuse(var_config, device, model=model)
            if ra_cfg.get("freeze_addresses", ra_cfg.get("freeze_router", True)):
                model.freeze_addresses()

        if base_meta:
            _copy_compatible_weights(base_model, model)
            if benchmark_seq_lens:
                with torch.no_grad():
                    n = min(base_model.pos_emb.num_embeddings, model.pos_emb.num_embeddings)
                    model.pos_emb.weight[:n] = base_model.pos_emb.weight[:n]
                _expand_position_embeddings(model, max(benchmark_seq_lens))

        # Verify all routing layers use routing attention
        if attn_type == "routing":
            n_routing_layers = sum(
                1 for b in model.blocks if type(b.attn).__name__ == "RoutingSparseAttention"
            )
            assert n_routing_layers == model.n_layers, (
                f"Expected routing at all {model.n_layers} layers, got {n_routing_layers}"
            )
            training_info[var_name] = {"routing_layers": n_routing_layers}

        if load_checkpoint_path is not None:
            ckpt = Path(load_checkpoint_path)
            if ckpt.exists():
                _load_state_dict_tolerant(model, ckpt, device)
                logger.info("Loaded checkpoint (tolerant): %s", ckpt)
            else:
                logger.warning("Checkpoint not found: %s", ckpt)

        # Fine-tune: fair mode trains ALL variants equally; skip only if fair_finetune=false and skip=true
        should_train = not skip_finetune and (
            fair_finetune or (not ra_cfg.get("skip", False) and attn_type == "routing")
        )
        if ra_cfg.get("skip", False) and not fair_finetune:
            should_train = False

        variant_ckpt_dir = runner.checkpoint_dir / var_name
        if should_train:
            freeze_router = ra_cfg.get("freeze_router", True) if attn_type == "routing" else False
            aux_weight = ra_cfg.get("routing_aux_weight", 0.0)

            if freeze_router and attn_type == "routing":
                model.freeze_router()

            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable, lr=ra_cfg.get("lr", 3e-4))

            ra_eval_every = resolve_eval_every(var_config, "routing_attention")
            val_batches = var_config.get("validation", {}).get("max_batches", 1)
            eval_fn = None
            if ra_eval_every > 0:
                eval_loader = get_eval_dataloader(var_config)

                def eval_fn(m, _loader=eval_loader, _device=device, _batches=val_batches):
                    return evaluate_lm(m, _loader, _device, max_batches=_batches)

            trainer = TransformerTrainer(
                model=model,
                optimizer=optimizer,
                device=device,
                runner=runner,
                metrics_logger=metrics_logger,
                logger=logger,
                max_steps=shared_max_steps,
                eval_every=ra_eval_every,
                save_every=ra_cfg.get("save_every", 5000),
                eval_fn=eval_fn,
                freeze_router=freeze_router,
                routing_aux_weight=aux_weight if attn_type == "routing" else 0.0,
                routing_aux_loss_type=ra_cfg.get("routing_aux_loss_type", "infonce"),
                routing_top_k=var_config["router"]["top_k"],
                routing_temperature=var_config["router"]["temperature"],
                teacher_model=teacher_model if aux_weight > 0 else None,
                routing_aux_layer=ra_cfg.get("routing_aux_layer", -1),
                aux_hidden_source=ra_cfg.get("aux_hidden_source", "student"),
                aux_attention_source=ra_cfg.get("aux_attention_source", "teacher"),
                use_amp=var_config.get("training", {}).get("use_amp", True),
                digit_loss_weight=var_config["transformer"].get("digit_loss_weight", 1.0),
            )
            orig_dir = runner.checkpoint_dir
            runner.checkpoint_dir = variant_ckpt_dir
            runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.runner = runner
            train_result = trainer.train(get_train_dataloader(var_config))
            training_info[var_name] = {
                **training_info.get(var_name, {}),
                **train_result,
                "trained": True,
                "freeze_router": freeze_router,
                "eval_every": ra_eval_every,
            }
            runner.checkpoint_dir = orig_dir
        else:
            training_info[var_name] = {"trained": False, "steps": 0}

        if use_best_checkpoint:
            best_path = variant_ckpt_dir / "best.pt"
            if best_path.exists():
                load_checkpoint(best_path, model, device=device, strict=False)
                training_info[var_name]["used_best_checkpoint"] = str(best_path)
                logger.info("Evaluating best checkpoint: %s", best_path)
            else:
                logger.warning("No best.pt at %s — using final weights", best_path)
                training_info[var_name]["used_best_checkpoint"] = None

        scaling_only = bool(benchmark_seq_lens and skip_finetune)
        if scaling_only:
            lm_results[var_name] = {
                "loss": None,
                "perplexity": None,
                "digit_accuracy": None,
                "skipped": "scaling_benchmark",
            }
            logger.info(
                "Skipping LM eval for scaling-only benchmark (device=%s%s)",
                device,
                f", gpu={torch.cuda.get_device_name(device)}" if device.type == "cuda" else "",
            )
        else:
            eval_metrics = evaluate_lm(
                model,
                get_eval_dataloader(var_config),
                device,
                max_batches=var_config["evaluation"]["max_batches"],
            )
            lm_results[var_name] = eval_metrics

        if benchmark_seq_lens:
            max_bench_seq = max(benchmark_seq_lens)
            patch_model_retrievers(model, max_bench_seq)
            bench_retrieval_cfg = sync_retrieval_config_dict(
                var_config.get("retrieval"), max_bench_seq
            )
            var_config["retrieval"] = bench_retrieval_cfg
            base_batch = var_config["data"]["batch_size"]
            seq_results = {}
            for seq_len in benchmark_seq_lens:
                seq_results[str(seq_len)] = benchmark_attention(
                    model,
                    seq_len=seq_len,
                    batch_size=_benchmark_batch_size(base_batch, seq_len),
                    device=device,
                    vocab_size=model.lm_head.out_features,
                    retrieval_cfg=bench_retrieval_cfg,
                    num_runs=bench_cfg["num_runs"],
                    num_warmup=bench_cfg["num_warmup"],
                )
            benchmark_results[var_name] = {"seq_len_sweep": seq_results}
        else:
            benchmark_results[var_name] = benchmark_attention(
                model,
                seq_len=var_config["data"]["seq_len"],
                batch_size=var_config["data"]["batch_size"],
                device=device,
                vocab_size=model.lm_head.out_features,
                retrieval_cfg=var_config.get("retrieval"),
                num_runs=bench_cfg["num_runs"],
                num_warmup=bench_cfg["num_warmup"],
            )

        if not scaling_only:
            if attn_type == "routing" and cache is not None:
                recall_results[var_name] = _eval_routing_recall(
                    model.router, cache, var_config, device, dry_run=dry_run
                )
            elif attn_type == "learned_address" and cache is not None and model.address_book is not None:
                recall_results[var_name] = _eval_routing_recall(
                    model.address_book, cache, var_config, device, dry_run=dry_run
                )

            save_checkpoint(
                runner.checkpoint_dir / f"{var_name}_final.pt",
                model,
                metrics=lm_results[var_name],
                extra={"variant": var_name, "attention_type": attn_type},
            )

    scaling_only_run = bool(benchmark_seq_lens and skip_finetune)
    if scaling_only_run:
        comparison = {"mode": "scaling_benchmark", "benchmark_seq_lens": benchmark_seq_lens}
        loss_delta = None
        verdict = "scaling"
    else:
        comparison = compare_attention_types(lm_results)
        primary_key = variants_to_run[0] if len(variants_to_run) == 1 else "routing"
        primary_metrics = lm_results.get(primary_key, {})
        primary_loss = primary_metrics.get("loss", float("inf"))
        dense_loss = lm_results.get("dense", {}).get("loss")
        max_allowed = config.get("success_criteria", {}).get("lm_loss_increase_max", 0.05)
        if primary_key == "dense":
            loss_delta = 0.0
            verdict = "baseline"
        elif dense_loss is not None:
            loss_delta = primary_loss - dense_loss
            verdict = "promising" if loss_delta <= max_allowed else "degraded"
        else:
            loss_delta = None
            verdict = "metrics_only"

    save_stats(runner.stats_dir, "lm_results", lm_results)
    save_stats(runner.stats_dir, "benchmark_results", benchmark_results)
    save_stats(runner.stats_dir, "recall_results", recall_results)
    save_stats(runner.stats_dir, "training_info", training_info)
    save_stats(runner.stats_dir, "comparison", comparison)
    save_stats(runner.stats_dir, "comparison_protocol", {
        "fair_finetune": fair_finetune,
        "shared_max_steps": shared_max_steps,
        "skip_finetune": skip_finetune,
        "use_best_checkpoint": use_best_checkpoint,
        "benchmark_seq_lens": benchmark_seq_lens,
        "same_init_checkpoint": "Experiment_1 transformer",
        "variants": variants_to_run,
    })

    primary_key = variants_to_run[0] if len(variants_to_run) == 1 else "routing"
    primary_metrics = lm_results.get(primary_key, {})
    if not scaling_only_run:
        save_stats(runner.stats_dir, "verdict", {
            "loss_delta": loss_delta,
            "assessment": verdict,
            "primary_variant": primary_key,
        })
    else:
        save_stats(runner.stats_dir, "verdict", {
            "assessment": verdict,
            "primary_variant": primary_key,
        })

    plot_paths = {}
    if not scaling_only_run:
        try:
            from routing_attention.evaluation.plots import save_all_plots

            plot_paths = save_all_plots(runner.plots_dir, {
                "lm_comparison": {
                    "loss": {k: v["loss"] for k, v in lm_results.items()},
                    "perplexity": {k: v["perplexity"] for k, v in lm_results.items()},
                },
                "metrics_history": metrics_logger._history,
            })
        except ImportError:
            plot_paths = {}
    metrics_logger.close()

    summary = {
        "experiment": "Experiment_4",
        "run_dir": str(runner.run_dir),
        "variant": variant or "compare_all",
        "attention_type": ATTENTION_VARIANTS.get(primary_key, primary_key),
        "lm_results": lm_results,
        "lm_loss": primary_metrics.get("loss"),
        "perplexity": primary_metrics.get("perplexity"),
        "digit_accuracy": primary_metrics.get("digit_accuracy"),
        "benchmark_results": benchmark_results,
        "recall_results": recall_results,
        "training_info": training_info,
        "comparison": comparison,
        "verdict": verdict,
        "loss_delta_vs_dense": None if scaling_only_run else loss_delta,
        "plot_paths": plot_paths,
    }
    runner.finalize(summary)
    if scaling_only_run:
        logger.info(
            "Experiment 4 complete. variant=%s scaling_benchmark verdict=%s",
            primary_key,
            verdict,
        )
    else:
        logger.info(
            "Experiment 4 complete. variant=%s lm_loss=%.4f digit_acc=%s verdict=%s",
            primary_key,
            primary_metrics.get("loss", float("nan")),
            primary_metrics.get("digit_accuracy"),
            verdict,
        )
    return summary


def _eval_routing_recall(
    router,
    cache: dict,
    config: dict,
    device,
    dry_run: bool = False,
) -> dict:
    top_k = config["router"]["top_k"]
    max_eval_samples, max_eval_tokens = get_recall_eval_limits(config, dry_run=dry_run)
    from routing_attention.models.learned_address import PerLayerAddressBook

    if "layers" in cache and isinstance(router, (PerLayerRouter, PerLayerAddressBook)):
        return evaluate_per_layer_recall(
            router=router,
            cache=cache,
            device=device,
            recall_k=top_k,
            n_layers=router.n_layers,
            max_eval_samples=max_eval_samples,
            max_eval_tokens=max_eval_tokens,
            show_progress=False,
        )
    last_li = max(cache["layers"].keys()) if "layers" in cache else 0
    ld = cache.get("layers", {}).get(last_li, cache)
    return compute_recall_from_router(
        ld["hidden_states"],
        ld["attention"],
        router,
        layer_idx=last_li if isinstance(router, PerLayerRouter) else None,
        k=top_k,
        max_samples=max_eval_samples,
        max_tokens=max_eval_tokens,
    )


def _apply_variant_config(config: dict, var_name: str) -> dict:
    from routing_attention.utils.config import apply_variant
    return apply_variant(config, var_name)


def _copy_compatible_weights(src: TransformerLM, dst: TransformerLM) -> None:
    """Copy shared weights (embeddings, FFN, QKV where shapes match) from dense init."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied, skipped = [], []
    for key in dst_state:
        if key in src_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key]
            copied.append(key)
        elif "router" not in key:
            skipped.append(key)
    dst.load_state_dict(dst_state, strict=False)
