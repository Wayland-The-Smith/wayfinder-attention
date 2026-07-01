"""

Experiment 1: Can routing reconstruct attention neighborhoods?



Trains per-layer routers by default (fixes layer mismatch).

Supports all loss types: infonce, infonce_sampled, mse, kl, multi_scale.



Task: learn a routing vector field r(h) such that top-k(r) ≈ top-k(attention)

from a frozen dense transformer on MNIST pixel sequences (784-token autoregressive LM).

Router trains on a TRAIN cache; Step D evaluates on a disjoint HOLDOUT cache.

"""



from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.optim as optim



from experiments.common import (

    announce_step,

    build_router,

    collect_router_attention_caches,

    ensure_address_book_on_model,

    load_addresses_from_reuse,

    get_recall_eval_limits,

    get_train_dataloader,

    init_experiment_runtime,

    load_experiment_config,

    load_router_from_reuse,

    load_transformer_from_reuse,

    make_router_eval_fn,

    make_transformer_eval_fn,

    resolve_cache_batch_counts,

    resolve_eval_every,

    resolve_router_training_layer_idx,

    router_cache_paths,

    save_stats,
    write_idea_manifest,
)

from routing_attention.evaluation.plots import save_all_plots

from routing_attention.evaluation.recall import (
    _normalize_eval_layer_indices,
    compute_recall_from_router,
    evaluate_key_vector_recall_from_cache,
    evaluate_learned_address_recall_from_cache,
    evaluate_per_layer_recall,
    filter_recall_metrics_by_layers,
)

from routing_attention.models.router import MultiScaleRouter, PerLayerRouter

from routing_attention.training.trainer import (

    RouterTrainer,

    TransformerTrainer,

    train_per_layer_addresses,

    train_per_layer_routers,

)

from routing_attention.utils.experiment import ExperimentRunner

from routing_attention.utils.logging import MetricsLogger, setup_logging





def run(

    variant: str | None = None,

    dry_run: bool = False,

    config_override: dict | None = None,

) -> dict[str, Any]:

    config = load_experiment_config(1, variant=variant, config_override=config_override)

    device = init_experiment_runtime(config)



    runner = ExperimentRunner(

        experiment_name=config["experiment"]["name"],

        config=config,

        dry_run=dry_run,

    )

    config = runner.config  # apply dry-run step overrides when enabled

    logger = setup_logging(runner.run_dir)

    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)

    logger.info("Starting Experiment 1: %s", config["experiment"]["description"])

    exp_cfg = config.get("experiment", {})
    idea_manifest = config.get("idea_manifest")
    if idea_manifest:
        manifest_path = write_idea_manifest(runner.run_dir, idea_manifest)
        logger.info("Wrote idea manifest: %s", manifest_path)

    router_cfg = config["router"]

    router_mode = router_cfg.get("mode", "per_layer")

    all_layers = router_cfg.get("mode", "per_layer") == "per_layer" or config["data_collection"].get("all_layers", True)

    per_head = config["data_collection"].get("per_head", False)

    train_batches, holdout_batches = resolve_cache_batch_counts(config)



    # --- Phase A: Train or load transformer ---

    announce_step(

        "A - Train dense transformer",

        logger,

        f"{config['transformer']['max_steps']} steps (or load checkpoint)",

    )

    model, ckpt_meta = load_transformer_from_reuse(config, device, attention_type="dense")



    if ckpt_meta is None:

        logger.info(
            "Phase A tasks: MNIST digit classification (10-class CE) + pixel LM auxiliary. "
            "Logged every step: phase_a/digit_loss, phase_a/digit_vs_random (vs ln(10)), "
            "phase_a/digit_accuracy, phase_a/lm_loss, phase_a/lm_vs_random. "
            "Phase C logs: phase_c/routing_loss, phase_c/routing_recall, phase_c/routing_vs_random."
        )

        optimizer = optim.AdamW(

            model.parameters(),

            lr=config["transformer"]["lr"],

            weight_decay=config["transformer"]["weight_decay"],

        )



        eval_every = resolve_eval_every(config, "transformer")

        eval_fn = make_transformer_eval_fn(config, device) if eval_every > 0 else None



        trainer = TransformerTrainer(

            model=model, optimizer=optimizer, device=device, runner=runner,

            metrics_logger=metrics_logger, logger=logger,

            max_steps=config["transformer"]["max_steps"],

            eval_every=eval_every,

            save_every=config["transformer"]["save_every"],

            eval_fn=eval_fn,

            use_amp=config.get("training", {}).get("use_amp", True),

            digit_loss_weight=config["transformer"].get("digit_loss_weight", 1.0),

        )

        orig_ckpt_dir = runner.checkpoint_dir

        runner.checkpoint_dir = orig_ckpt_dir / "transformer"

        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer.runner = runner

        train_result = trainer.train(get_train_dataloader(config))

        save_stats(runner.stats_dir, "transformer_training", train_result)

        runner.checkpoint_dir = orig_ckpt_dir

    else:
        logger.info("Reused transformer checkpoint: step %s", ckpt_meta.get("step"))

    if exp_cfg.get("phase_a2_joint_aux") and ckpt_meta is not None:
        ra_cfg = config.get("routing_attention", {})
        a2_steps = int(exp_cfg.get("phase_a2_steps", 2000))
        announce_step("A2 - Joint aux routing fine-tune", logger, f"{a2_steps} steps, λ={ra_cfg.get('routing_aux_weight', 0.1)}")
        joint_router = build_router(config).to(device)
        model.router = joint_router
        teacher = deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        for p in model.parameters():
            p.requires_grad = True
        joint_opt = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=ra_cfg.get("lr", config["transformer"]["lr"]),
        )
        a2_trainer = TransformerTrainer(
            model=model,
            optimizer=joint_opt,
            device=device,
            runner=runner,
            metrics_logger=metrics_logger,
            logger=logger,
            max_steps=a2_steps,
            eval_every=0,
            save_every=a2_steps,
            freeze_router=False,
            routing_aux_weight=ra_cfg.get("routing_aux_weight", 0.1),
            routing_aux_loss_type=ra_cfg.get("routing_aux_loss_type", "infonce"),
            routing_top_k=router_cfg["top_k"],
            routing_temperature=router_cfg["temperature"],
            teacher_model=teacher,
            routing_aux_layer=ra_cfg.get("routing_aux_layer", -1),
            aux_hidden_source=ra_cfg.get("aux_hidden_source", "student"),
            use_amp=config.get("training", {}).get("use_amp", True),
            digit_loss_weight=config["transformer"].get("digit_loss_weight", 1.0),
        )
        orig_ckpt_dir = runner.checkpoint_dir
        runner.checkpoint_dir = orig_ckpt_dir / "joint_aux"
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        a2_trainer.runner = runner
        save_stats(runner.stats_dir, "joint_aux_training", a2_trainer.train(get_train_dataloader(config)))
        runner.checkpoint_dir = orig_ckpt_dir

    for p in model.parameters():
        p.requires_grad = False
    model.eval()



    # --- Phase B: Collect train + holdout attention caches ---

    announce_step(

        "B - Collect attention caches (train + holdout)",

        logger,

        f"train={train_batches} batches, holdout={holdout_batches} batches, all_layers={all_layers}",

    )

    cache_force_refresh = bool(exp_cfg.get("phase_a2_joint_aux") and ckpt_meta is not None)

    train_cache_path, holdout_cache_path = collect_router_attention_caches(

        model=model,

        config=config,

        runner=runner,

        device=device,

        logger=logger,

        all_layers=all_layers,

        per_head=per_head,

        force_refresh=cache_force_refresh,

    )

    save_stats(runner.stats_dir, "cache_splits", {

        "train_cache": str(train_cache_path),

        "holdout_cache": str(holdout_cache_path),

        "train_batches": train_batches,

        "holdout_batches": holdout_batches,

        "batch_size": config["data_collection"].get("batch_size", config["data"]["batch_size"]),

    })



    # --- Phase C: Train router or learned addresses on TRAIN cache only ---
    skip_router_training = bool(exp_cfg.get("skip_router_training", False))
    eval_mode = exp_cfg.get("eval_mode", "router_mlp")
    train_learned_addresses = eval_mode == "learned_address" or bool(
        exp_cfg.get("train_learned_addresses", False)
    )
    router = None
    router_meta = None
    address_book = None
    address_meta = None
    router_result = {}

    if skip_router_training and eval_mode == "key_vector":
        announce_step("C - Skip router training", logger, "key_vector baseline (Q/K projections only)")
    elif train_learned_addresses:
        announce_step(
            "C - Train learned addresses",
            logger,
            f"per_layer, {router_cfg['max_steps']} steps/layer, train cache only",
        )
        address_book, address_meta = load_addresses_from_reuse(config, device, model=model)
    elif skip_router_training and exp_cfg.get("phase_a2_joint_aux") and getattr(model, "router", None) is not None:
        announce_step("C - Skip router training", logger, "using jointly trained router from Phase A2")
        router = model.router
    elif skip_router_training:
        announce_step("C - Skip router training", logger, exp_cfg.get("skip_reason", "experiment config"))
    else:
        announce_step(
            "C - Train router",
            logger,
            f"mode={router_mode}, {router_cfg['max_steps']} steps per layer, train cache only",
        )
        router, router_meta = load_router_from_reuse(config, device)

    if train_learned_addresses and address_meta is None:

        router_eval_every = resolve_eval_every(config, "router")

        logger.info(
            "Phase C active task: learned addresses (InfoNCE vs dense attention neighborhoods). "
            "Train %d steps/layer; every %d steps eval holdout Recall@K and save best.pt. "
            "Logged every step: phase_c/routing_loss, phase_c/routing_recall, phase_c/routing_vs_random. "
            "similarity=%s loss_type=%s",
            router_cfg["max_steps"],
            router_eval_every,
            config.get("learned_address", {}).get("similarity", "asymmetric"),
            router_cfg["loss_type"],
        )

        address_optimizer = optim.AdamW(address_book.parameters(), lr=router_cfg["lr"])
        top_k = router_cfg["top_k"]
        orig_ckpt_dir = runner.checkpoint_dir
        runner.checkpoint_dir = orig_ckpt_dir / "addresses"
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        eval_max_samples = config.get("validation", {}).get("eval_max_samples", 8)
        eval_fn = make_router_eval_fn(top_k, eval_max_samples=eval_max_samples) if router_eval_every > 0 else None

        router_result = train_per_layer_addresses(
            address_book=address_book,
            cache_path=train_cache_path,
            holdout_cache_path=holdout_cache_path,
            optimizer=address_optimizer,
            device=device,
            runner=runner,
            metrics_logger=metrics_logger,
            logger=logger,
            config=config,
            eval_fn=eval_fn,
        )
        save_stats(runner.stats_dir, "address_training", router_result)
        runner.checkpoint_dir = orig_ckpt_dir

    elif not skip_router_training and router_meta is None:

        router_eval_every = resolve_eval_every(config, "router")

        logger.info(
            "Phase C active task: routing (InfoNCE vs dense attention neighborhoods). "
            "Train %d steps/layer; every %d steps eval holdout Recall@K and save best.pt. "
            "Logged every step: phase_c/routing_loss, phase_c/routing_recall, phase_c/routing_vs_random. "
            "mode=%s loss_type=%s",
            router_cfg["max_steps"],
            router_eval_every,
            router_mode,
            router_cfg["loss_type"],
        )

        router_optimizer = optim.AdamW(router.parameters(), lr=router_cfg["lr"])

        top_k = router_cfg["top_k"]



        orig_ckpt_dir = runner.checkpoint_dir

        runner.checkpoint_dir = orig_ckpt_dir / "router"

        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        eval_max_samples = config.get("validation", {}).get("eval_max_samples", 8)

        if isinstance(router, PerLayerRouter):

            eval_fn = make_router_eval_fn(top_k, eval_max_samples=eval_max_samples) if router_eval_every > 0 else None

            router_result = train_per_layer_routers(

                router=router,

                cache_path=train_cache_path,

                holdout_cache_path=holdout_cache_path,

                optimizer=router_optimizer,

                device=device,

                runner=runner,

                metrics_logger=metrics_logger,

                logger=logger,

                config=config,

                eval_fn=eval_fn,

            )

        else:

            eval_fn = make_router_eval_fn(top_k, eval_max_samples=eval_max_samples) if router_eval_every > 0 else None

            router_trainer = RouterTrainer(

                router=router,

                optimizer=router_optimizer,

                device=device,

                runner=runner,

                metrics_logger=metrics_logger,

                logger=logger,

                loss_type=router_cfg["loss_type"],

                top_k=top_k,

                temperature=router_cfg["temperature"],

                max_steps=router_cfg["max_steps"],

                eval_every=router_eval_every,

                save_every=router_cfg["save_every"],

                eval_fn=eval_fn,

                batch_size=router_cfg.get("batch_size", 4),

                num_negatives=router_cfg.get("num_negatives", 64),

                layer_idx=resolve_router_training_layer_idx(config),

                attention_supervision=config["data_collection"].get("attention_supervision", "head_avg"),

                eval_max_samples=eval_max_samples,

                use_amp=config.get("training", {}).get("use_amp", True),

            )

            router_result = router_trainer.train_on_cache(train_cache_path, holdout_cache_path)



        save_stats(runner.stats_dir, "router_training", router_result)

        runner.checkpoint_dir = orig_ckpt_dir

    elif train_learned_addresses:
        logger.info("Reused learned-address checkpoint")
        router_result = {}
    elif not skip_router_training:
        logger.info("Reused router checkpoint")
        router_result = {}



    # --- Phase D: Final evaluation on HOLDOUT cache only ---

    eval_cfg = config.get("evaluation", {})

    max_eval_samples, max_eval_tokens = get_recall_eval_limits(config, dry_run=dry_run)

    include_mean_rank = eval_cfg.get("include_mean_rank", False)

    announce_step(

        "D - Final Recall@K evaluation (holdout)",

        logger,

        f"k={eval_cfg.get('recall_k', router_cfg['top_k'])}, "

        f"samples={max_eval_samples or 'all holdout'}, tokens={max_eval_tokens or 'all'}",

    )

    from routing_attention.data.chunked_cache import (
        cache_is_ready,
        compute_random_baseline_from_cache,
        evaluate_per_layer_recall_from_cache,
        is_chunked_cache,
    )

    recall_k = eval_cfg.get("recall_k", router_cfg["top_k"])
    eval_layers_cfg = eval_cfg.get("eval_layers")
    layer_indices = None
    if eval_layers_cfg is not None:
        layer_indices = _normalize_eval_layer_indices(list(eval_layers_cfg), config["model"]["n_layers"])

    if eval_mode == "key_vector":
        recall_metrics = evaluate_key_vector_recall_from_cache(
            model=model,
            cache_path=holdout_cache_path,
            device=device,
            recall_k=recall_k,
            n_layers=config["model"]["n_layers"],
            max_eval_samples=max_eval_samples,
            max_eval_tokens=max_eval_tokens,
            show_progress=not dry_run,
            layer_indices=layer_indices,
        )
        recall_val = recall_metrics["mean_recall"]
    elif eval_mode == "learned_address":
        if address_book is None:
            address_book, _ = load_addresses_from_reuse(config, device, model=model)
        if address_book is None:
            address_book = ensure_address_book_on_model(model, config, device=device)
        recall_metrics = evaluate_learned_address_recall_from_cache(
            address_book=address_book,
            cache_path=holdout_cache_path,
            device=device,
            recall_k=recall_k,
            n_layers=config["model"]["n_layers"],
            max_eval_samples=max_eval_samples,
            max_eval_tokens=max_eval_tokens,
            show_progress=not dry_run,
            layer_indices=layer_indices,
        )
        recall_val = recall_metrics["mean_recall"]
    else:
        if router is None:
            raise RuntimeError(
                "Phase D requires a router unless eval_mode is key_vector or learned_address"
            )

        use_chunked = is_chunked_cache(holdout_cache_path)
        use_per_layer = isinstance(router, PerLayerRouter) and (
            use_chunked or cache_is_ready(holdout_cache_path)
        )

        if isinstance(router, PerLayerRouter) and use_per_layer:
            if use_chunked:
                recall_metrics = evaluate_per_layer_recall_from_cache(
                    router=router,
                    cache_path=holdout_cache_path,
                    device=device,
                    recall_k=recall_k,
                    n_layers=router.n_layers,
                    max_eval_samples=max_eval_samples,
                    max_eval_tokens=max_eval_tokens,
                    include_mean_rank=include_mean_rank,
                    show_progress=not dry_run,
                )
            else:
                holdout_cache = torch.load(holdout_cache_path, map_location="cpu", weights_only=False)
                recall_metrics = evaluate_per_layer_recall(
                    router=router,
                    cache=holdout_cache,
                    device=device,
                    recall_k=recall_k,
                    n_layers=router.n_layers,
                    max_eval_samples=max_eval_samples,
                    max_eval_tokens=max_eval_tokens,
                    include_mean_rank=include_mean_rank,
                    show_progress=not dry_run,
                )
            recall_val = recall_metrics["mean_recall"]
        else:
            from routing_attention.data.chunked_cache import load_monolithic_or_layer

            hidden, attention = load_monolithic_or_layer(
                holdout_cache_path, 0, device, max_samples=max_eval_samples,
            )
            router.eval()
            recall_metrics = compute_recall_from_router(
                hidden,
                attention,
                router,
                k=recall_k,
                max_samples=max_eval_samples,
                max_tokens=max_eval_tokens,
                include_mean_rank=include_mean_rank,
            )
            recall_val = recall_metrics.get(f"recall@{min(recall_k, hidden.shape[1])}", 0)

    if layer_indices is not None:
        recall_metrics = filter_recall_metrics_by_layers(recall_metrics, layer_indices)
        recall_val = recall_metrics["mean_recall"]

    random_baseline = compute_random_baseline_from_cache(
        holdout_cache_path,
        device,
        recall_k=recall_k,
        max_eval_samples=max_eval_samples,
        max_eval_tokens=max_eval_tokens,
        per_layer=True,
        n_layers=config["model"]["n_layers"],
    )
    if layer_indices is not None and isinstance(random_baseline, dict):
        random_baseline = filter_recall_metrics_by_layers(
            {"per_layer": random_baseline.get("per_layer", {}), "mean_recall": random_baseline.get("mean_recall", 0)},
            layer_indices,
        )

    recall_metrics["eval_split"] = "holdout"

    recall_metrics["random_baseline"] = random_baseline

    recall_metrics["recall_above_random"] = recall_val - random_baseline.get("mean_recall", 0)



    save_stats(runner.stats_dir, "recall_metrics", recall_metrics)

    criteria = config.get("success_criteria", {}).get("recall_at_32", {})

    verdict = _assess_recall(recall_val, criteria, dry_run=dry_run, above_random=recall_metrics["recall_above_random"])

    save_stats(runner.stats_dir, "verdict", {

        "recall": recall_val,

        "random_baseline": random_baseline.get("mean_recall", 0),

        "recall_above_random": recall_metrics["recall_above_random"],

        "assessment": verdict,

        "dry_run": dry_run,

        "eval_split": "holdout",

    })



    plot_paths = save_all_plots(runner.plots_dir, {"metrics_history": metrics_logger._history})

    metrics_logger.close()



    summary = {

        "experiment": "Experiment_1",

        "run_dir": str(runner.run_dir),

        "variant": variant or "default",

        "router_mode": router_mode,

        "recall_metrics": recall_metrics,

        "verdict": verdict,

        "plot_paths": plot_paths,

        "transformer_checkpoint": str(runner.checkpoint_dir / "transformer" / "best.pt"),

        "router_checkpoint": str(runner.checkpoint_dir / "router" / "best.pt"),

        "train_attention_cache": str(train_cache_path),

        "holdout_attention_cache": str(holdout_cache_path),

    }

    runner.finalize(summary)

    logger.info(

        "Experiment 1 complete. Verdict: %s (holdout Recall=%.3f, random=%.3f, delta=%+.3f)",

        verdict,

        recall_val,

        random_baseline.get("mean_recall", 0),

        recall_metrics["recall_above_random"],

    )

    return summary





def _assess_recall(

    recall: float,

    criteria: dict,

    dry_run: bool = False,

    above_random: float = 0.0,

) -> str:

    if dry_run:

        return "smoke_test_only"

    if not criteria:

        return "no_criteria"

    if recall >= criteria.get("breakthrough", 0.9):

        return "breakthrough"

    if recall >= criteria.get("strong", 0.8):

        return "strong"

    if recall >= criteria.get("interesting", 0.5) and above_random >= 0.15:

        return "interesting"

    if recall >= criteria.get("dead", 0.2) or above_random >= 0.08:

        return "weak_signal"

    return "likely_dead"





if __name__ == "__main__":

    run(dry_run=True)


