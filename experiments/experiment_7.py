"""
Experiment 7 — procedural long-context retrieval benchmark.

One fixed context length per sub-experiment: train all variants at T, then compare
on the held-out grid filtered to that T (shared across variants).

Training protocol (``long_context_benchmark.training_protocol: two_stage``):
  Stage A — train ``dense_flash`` on NIAH at fixed T → checkpoint C_dense(T).
  Stage B — every other variant loads C_dense(T), then fine-tunes on NIAH using
            its own attention mechanism (sparse top-k for routing variants;
            router/addresses frozen when configured).
"""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.common import (
    announce_step,
    build_router,
    build_transformer,
    init_experiment_runtime,
    load_experiment_config,
    set_seed,
)
from experiments.experiment_4 import (
    ATTENTION_VARIANTS,
    _apply_variant_config,
    _copy_compatible_weights,
    _expand_position_embeddings,
    _load_state_dict_tolerant,
)
from routing_attention.benchmarks.long_context import (
    LongContextBenchmarkConfig,
    LongContextEvaluator,
    save_all_benchmark_plots,
)
from routing_attention.benchmarks.long_context.dataset import (
    get_long_context_dataloader,
    transfer_batch_to_device,
)
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
    production_manifest_for_variants,
)
from routing_attention.benchmarks.long_context.routing_setup import apply_routing_variant_settings
from routing_attention.benchmarks.long_context.runtime import (
    assert_expected_device,
    collect_device_info,
    peak_vram_mb,
    reset_peak_vram,
    verify_model_on_device,
)
from routing_attention.models.fast_attention import (
    backend_status,
    warmup_fla_linear_kernels,
    warmup_flex_sliding_window,
)
from routing_attention.models.transformer import TransformerLM
from routing_attention.utils.checkpoint import save_checkpoint, load_checkpoint
from routing_attention.utils.experiment import ExperimentRunner, resolve_checkpoint_path
from routing_attention.utils.logging import MetricsLogger, setup_logging

SPARSE_ROUTING_VARIANTS = frozenset(
    {"routing_asymmetric", "learned_address_k32", "key_vector_k32"}
)

ROUTING_INDEX_FROZEN_VARIANTS = frozenset({"routing_asymmetric"})

EXP7_ATTENTION_VARIANTS = {
    **ATTENTION_VARIANTS,
    "dense_flash": "dense_flash",
    "linear": "linear",
    "local_window64": "local",
    "local_window256": "local",
}


def _expand_needle_scatter_curriculum(
    spec: dict[str, Any],
    *,
    train_context_length: int | None = None,
) -> list[dict[str, Any]]:
    """Build ``needle_scatter_curriculum`` from a compact ramp spec."""
    start_max = int(spec.get("start_max", 50))
    step_chars = int(spec.get("step_chars", 50))
    steps_per_tier = int(spec.get("steps_per_tier", 750))
    haystack_len = spec.get("haystack_len")
    if haystack_len is None:
        if train_context_length is None:
            raise ValueError(
                "needle_scatter_curriculum_spec requires haystack_len or train_context_length"
            )
        haystack_len = train_context_length - int(spec.get("suffix_budget", 20))
    haystack_len = int(haystack_len)

    curriculum: list[dict[str, Any]] = []
    max_pos = start_max
    until = steps_per_tier
    while max_pos < haystack_len:
        curriculum.append(
            {
                "until_step": until,
                "scatter_placement_min": 0,
                "scatter_placement_max": max_pos,
            }
        )
        max_pos += step_chars
        until += steps_per_tier

    curriculum.append(
        {
            "until_step": until,
            "scatter_placement_min": 0,
            "scatter_placement_max": None,
        }
    )
    return curriculum


def _apply_scatter_curriculum_spec(config: dict, train_context_length: int) -> None:
    bench = config.setdefault("long_context_benchmark", {})
    spec = bench.get("needle_scatter_curriculum_spec")
    if not spec:
        return
    if bench.get("needle_scatter_curriculum"):
        return
    bench["needle_scatter_curriculum"] = _expand_needle_scatter_curriculum(
        spec,
        train_context_length=train_context_length,
    )


def _benchmark_config_from_run(config: dict) -> LongContextBenchmarkConfig:
    return LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))


def _effective_train_batch_size(
    batch_size: int,
    train_context_length: int,
    attn_type: str,
) -> int:
    """Cap batch size so long-context dense training fits on ~32GB GPUs."""
    bs = max(1, batch_size)
    if attn_type not in ("dense", "dense_flash"):
        if train_context_length >= 32768:
            return min(bs, 2)
        return bs
    if train_context_length >= 32768:
        return 1
    if train_context_length >= 16384:
        return min(bs, 1)
    if train_context_length >= 8192:
        return min(bs, 2)
    return bs


def _training_protocol(config: dict) -> str:
    return str(
        config.get("long_context_benchmark", {}).get("training_protocol", "two_stage")
    )


def _is_two_stage(config: dict) -> bool:
    return _training_protocol(config) == "two_stage"


def _curriculum_context_length(step: int, base: int, curriculum: list[dict]) -> int:
    """Resolve train T from optional ``context_curriculum`` milestones."""
    if not curriculum:
        return base
    ordered = sorted(curriculum, key=lambda e: int(e["until_step"]))
    for entry in ordered:
        if step < int(entry["until_step"]):
            return int(entry["context_length"])
    return int(ordered[-1]["context_length"])


_SUFFIX_CURRICULUM_KEYS = frozenset(
    {
        "suffix_placement",
        "suffix_depth_min",
        "suffix_depth_max",
        "suffix_after_needles_gap_max",
        "synthetic_decoy_keys",
    }
)


_NEEDLE_SCATTER_CURRICULUM_KEYS = frozenset(
    {
        "scatter_placement_min",
        "scatter_placement_max",
    }
)


_SYNTHETIC_TASK_CURRICULUM_KEYS = frozenset(
    {
        "synthetic_hop_count",
        "synthetic_hop_count_min",
        "synthetic_hop_count_max",
        "num_distractors",
        "synthetic_decoy_addrs",
        "num_kv_pairs",
    }
)


def _curriculum_suffix_settings(step: int, curriculum: list[dict]) -> dict[str, Any]:
    """Resolve train suffix/decoy settings from optional ``suffix_curriculum``."""
    if not curriculum:
        return {}
    ordered = sorted(curriculum, key=lambda e: int(e["until_step"]))
    for entry in ordered:
        if step < int(entry["until_step"]):
            return {k: v for k, v in entry.items() if k != "until_step"}
    return {k: v for k, v in ordered[-1].items() if k != "until_step"}


def _curriculum_needle_scatter_settings(step: int, curriculum: list[dict]) -> dict[str, Any]:
    """Resolve needle placement bounds from optional ``needle_scatter_curriculum``."""
    if not curriculum:
        return {}
    ordered = sorted(curriculum, key=lambda e: int(e["until_step"]))
    for entry in ordered:
        if step < int(entry["until_step"]):
            return {k: v for k, v in entry.items() if k != "until_step"}
    return {k: v for k, v in ordered[-1].items() if k != "until_step"}


def _curriculum_synthetic_task_settings(step: int, curriculum: list[dict]) -> dict[str, Any]:
    """Resolve hop count / distractor settings from ``synthetic_task_curriculum``."""
    if not curriculum:
        return {}
    ordered = sorted(curriculum, key=lambda e: int(e["until_step"]))
    for entry in ordered:
        if step < int(entry["until_step"]):
            return {k: v for k, v in entry.items() if k != "until_step"}
    return {k: v for k, v in ordered[-1].items() if k != "until_step"}


def _apply_synthetic_hop_settings(bench_cfg: Any, hop_count: int) -> None:
    bench_cfg.synthetic_hop_count = hop_count
    bench_cfg.synthetic_hop_count_min = hop_count
    bench_cfg.synthetic_hop_count_max = hop_count


def _weighted_long_context_loss(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Token CE with optional per-position weights (answer tokens weighted higher)."""
    vocab = shift_logits.size(-1)
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    if shift_weights is None:
        mask = (shift_labels != -100).float()
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    mask = (shift_labels != -100).float()
    w = shift_weights * mask
    return (per_token * w).sum() / w.sum().clamp(min=1.0)


def _batch_uses_query_only_answer(meta_list: list[dict[str, Any]] | None) -> bool:
    if not meta_list:
        return False
    return all(m.get("label_mode") == "query_only_answer" for m in meta_list)


def _query_only_answer_aligned_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor | None,
    meta_list: list[dict[str, Any]],
) -> torch.Tensor:
    """
    CE at ``question_index`` — same logits position as ``score_query_only_answer`` eval.

    Causal LM shift would supervise ``logits[question_index - 1]``, which cannot see the
    query token at ``question_index``.
    """
    batch_size = logits.size(0)
    device = logits.device
    batch_idx = torch.arange(batch_size, device=device)
    question_index = torch.tensor(
        [int(m["question_index"]) for m in meta_list],
        device=device,
        dtype=torch.long,
    )
    pos_logits = logits[batch_idx, question_index]
    pos_labels = labels[batch_idx, question_index]
    per_example = torch.nn.functional.cross_entropy(pos_logits, pos_labels, reduction="none")
    if loss_weights is None:
        return per_example.mean()
    weights = loss_weights[batch_idx, question_index]
    return (per_example * weights).sum() / weights.sum().clamp(min=1.0)


def _pointer_batch_tensors(
    meta_list: list[dict[str, Any]] | None,
    device: torch.device,
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not meta_list:
        return None
    if not all(m.get("label_mode") == "pointer_index" for m in meta_list):
        return None
    question_index = torch.tensor(
        [int(m["question_index"]) for m in meta_list],
        device=device,
        dtype=torch.long,
    )
    output_head = getattr(model, "output_head", "lm_token")
    pointer_target_mode = getattr(model, "pointer_target_mode", "full_sequence")
    if output_head == "pointer_mlp" and pointer_target_mode == "value_slots":
        pointer_target = torch.tensor(
            [int(m["pointer_target_slot"]) for m in meta_list],
            device=device,
            dtype=torch.long,
        )
    else:
        pointer_target = torch.tensor(
            [int(m["pointer_target_index"]) for m in meta_list],
            device=device,
            dtype=torch.long,
        )
    return question_index, pointer_target


def _uses_pointer_output_head(model: torch.nn.Module) -> bool:
    return getattr(model, "output_head", "lm_token") in ("pointer_index", "pointer_mlp")


def _uses_pool_mlp_token_head(model: torch.nn.Module) -> bool:
    return getattr(model, "output_head", "lm_token") == "pool_mlp_token"


def _question_index_batch_tensor(
    meta_list: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [int(m["question_index"]) for m in meta_list],
        device=device,
        dtype=torch.long,
    )


def _query_only_answer_target_tensor(
    meta_list: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            int(m.get("query_only_answer_token") or m.get("pointer_target_token"))
            for m in meta_list
        ],
        device=device,
        dtype=torch.long,
    )


def _train_batch_gate_accuracy(
    logits: torch.Tensor,
    meta_list: list[dict[str, Any]] | None,
    evaluator: LongContextEvaluator,
    bench_cfg: LongContextBenchmarkConfig,
) -> tuple[float, int, int]:
    """Exact-match gate accuracy on the current train batch (same metric as holdout eval)."""
    if not meta_list:
        return 0.0, 0, 0
    gate_types = set(bench_cfg.primary_gate_task_types())
    correct = 0
    total = 0
    for batch_idx, meta in enumerate(meta_list):
        record = evaluator.score_sample(logits[batch_idx : batch_idx + 1], meta)
        if record.task_type in gate_types:
            total += 1
            if record.correct:
                correct += 1
    if total == 0:
        total = len(meta_list)
        for batch_idx, meta in enumerate(meta_list):
            record = evaluator.score_sample(logits[batch_idx : batch_idx + 1], meta)
            if record.correct:
                correct += 1
    acc = correct / total if total else 0.0
    return acc, correct, total


def _resolve_train_steps(config: dict, training_stage: str) -> int:
    train_cfg = config.get("transformer", {})
    default = int(train_cfg.get("max_steps", 0))
    if training_stage == "dense_pretrain":
        return int(train_cfg.get("dense_pretrain_steps") or default)
    if training_stage == "finetune_from_dense":
        ra_cfg = config.get("routing_attention", {})
        return int(
            train_cfg.get("sparse_finetune_steps")
            or ra_cfg.get("max_steps")
            or default
        )
    return default


def _create_shared_base_model(
    config: dict,
    max_seq_len: int,
) -> TransformerLM | None:
    """Legacy random shared init — only when not using two-stage dense checkpoints."""
    if _is_two_stage(config):
        return None
    if not bool(config.get("long_context_benchmark", {}).get("shared_init", True)):
        return None
    set_seed(config.get("seed", 42))
    base_config = deepcopy(config)
    base_config["model"]["max_seq_len"] = max_seq_len
    base = build_transformer(base_config, attention_type="dense").cpu()
    _expand_position_embeddings(base, max_seq_len)
    return base


def _save_dense_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    train_context_length: int,
    trained_steps: int,
) -> str:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        checkpoint_path,
        model,
        step=trained_steps,
        extra={
            "variant": "dense_flash",
            "attention_type": "dense_flash",
            "train_context_length": train_context_length,
            "training_stage": "dense_pretrain",
        },
    )
    return str(checkpoint_path)


def _verify_staged_training_protocol(
    variant: str,
    stage: str,
    routing_info: dict[str, Any],
    model: torch.nn.Module,
    *,
    two_stage: bool,
) -> dict[str, Any]:
    """
    Assert routing variants follow staged-training rules before optimization starts.
    Returns a small audit dict logged into run results.
    """
    audit: dict[str, Any] = {
        "variant": variant,
        "stage": stage,
        "two_stage": two_stage,
        "checks_passed": True,
    }
    errors: list[str] = []

    attn_type = routing_info.get("attn_type", "")
    if stage == "dense_pretrain":
        from_scratch = not routing_info.get("dense_init")
        if (
            not from_scratch
            and variant != "dense_flash"
            and attn_type not in ("dense", "dense_flash")
        ):
            errors.append(f"Stage A expects dense_flash, got {variant}/{attn_type}")
    elif stage == "finetune_from_dense":
        if not routing_info.get("dense_init"):
            errors.append("Stage B requires dense_init checkpoint path")
        if variant == "routing_asymmetric":
                if not routing_info.get("task_index_checkpoint"):
                    errors.append(
                        "routing_asymmetric requires NIAH task router index (Stage A.5); "
                        "missing task_index_checkpoint"
                    )
                n_train = routing_info.get("router_params_trainable")
                if n_train not in (0, None):
                    errors.append(f"router must be frozen during Stage B, got {n_train} trainable params")
                if routing_info.get("freeze_router") is not True:
                    errors.append("freeze_router must be true for routing_asymmetric Stage B")
        if variant == "learned_address_k32":
            n_train = routing_info.get("address_params_trainable", 0)
            if not n_train or n_train <= 0:
                errors.append("learned_address addresses must be trainable during NIAH Stage B fine-tune")
            if routing_info.get("freeze_addresses") is True:
                errors.append("freeze_addresses must be false for learned_address_k32 Stage B")
        if variant == "key_vector_k32":
            if routing_info.get("task_index_checkpoint"):
                errors.append("key_vector_k32 must not load a router index checkpoint")
            if routing_info.get("address_index_checkpoint"):
                errors.append("key_vector_k32 must not load an address index checkpoint")
            n_sparse = sum(
                1 for b in model.blocks if type(b.attn).__name__ == "KeyVectorSparseAttention"
            )
            if n_sparse != model.n_layers:
                errors.append(
                    f"key_vector_k32 expected KeyVectorSparseAttention at all layers, got {n_sparse}"
                )
        if variant == "linear":
            n_linear = sum(1 for b in model.blocks if type(b.attn).__name__ == "LinearAttention")
            if n_linear != model.n_layers:
                errors.append(f"linear expected LinearAttention at all layers, got {n_linear}")
        if variant in ("local_window64", "local_window256"):
            expected_w = 64 if variant == "local_window64" else 256
            n_local = sum(1 for b in model.blocks if type(b.attn).__name__ == "LocalAttention")
            if n_local != model.n_layers:
                errors.append(
                    f"{variant} expected LocalAttention at all layers, got {n_local}"
                )
            elif int(getattr(model.blocks[0].attn, "window_size", 0)) != expected_w:
                errors.append(f"{variant} expected window_size={expected_w}")
        if variant in ("learned_address_k32", "key_vector_k32"):
            if not routing_info.get("retrievers_patched"):
                errors.append("sparse variant missing retriever patch for long context")
            n_sparse = sum(
                1
                for b in model.blocks
                if type(b.attn).__name__
                in ("RoutingSparseAttention", "KeyVectorSparseAttention", "LearnedAddressSparseAttention")
            )
            if n_sparse != model.n_layers:
                errors.append(f"expected sparse attention at all {model.n_layers} layers, got {n_sparse}")
            audit["sparse_layers"] = n_sparse
    elif stage == "eval_only" and two_stage and variant != "dense_flash":
        if not routing_info.get("dense_init"):
            errors.append("eval_only Stage B variant missing dense_init")

    audit["trainable_params"] = routing_info.get("trainable_params")
    audit["total_params"] = routing_info.get("total_params")
    audit["dense_init"] = routing_info.get("dense_init")
    audit["freeze_router"] = routing_info.get("freeze_router")
    audit["freeze_addresses"] = routing_info.get("freeze_addresses")

    if errors:
        audit["checks_passed"] = False
        audit["errors"] = errors
        raise RuntimeError(
            f"Staged-training verification failed for {variant} (stage={stage}): " + "; ".join(errors)
        )
    return audit


def _build_variant_model(
    config: dict,
    variant: str,
    device: torch.device,
    max_seq_len: int,
    shared_base: TransformerLM | None = None,
    dense_checkpoint: Path | str | None = None,
    router_index_checkpoint: Path | str | None = None,
) -> tuple[torch.nn.Module, dict]:
    var_config = _apply_variant_config(deepcopy(config), variant)
    var_config["model"]["max_seq_len"] = max_seq_len
    model_cfg = var_config.get("model", {})
    attn_type = EXP7_ATTENTION_VARIANTS.get(variant) or model_cfg.get("attention_type") or "routing"
    var_config.setdefault("model", {})["attention_type"] = attn_type

    init_info: dict[str, Any] = {"variant": variant, "attn_type": attn_type}

    if attn_type == "routing":
        router = build_router(var_config).to(device)
        if router_index_checkpoint and Path(router_index_checkpoint).exists():
            load_checkpoint(router_index_checkpoint, router, device=device, strict=False)
            init_info["task_index_checkpoint"] = str(router_index_checkpoint)
            init_info["index_source"] = "niah_dense_teacher"
        model = build_transformer(var_config, attention_type="routing", router=router).to(device)
    else:
        model = build_transformer(var_config, attention_type=attn_type).to(device)

    if dense_checkpoint is not None:
        init_info["dense_init"] = str(dense_checkpoint)
    elif shared_base is not None:
        _copy_compatible_weights(shared_base, model)
        init_info["shared_init"] = True

    ckpt_map = config.get("reuse", {}).get("variant_checkpoints", {})
    ckpt_rel = ckpt_map.get(variant)
    if ckpt_rel:
        ckpt = resolve_checkpoint_path(ckpt_rel)
        if ckpt and Path(ckpt).exists():
            _load_state_dict_tolerant(model, Path(ckpt), device)
            init_info["variant_checkpoint"] = str(ckpt)

    if dense_checkpoint is not None:
        # Load trained dense trunk + compatible attention weights after variant build.
        _load_state_dict_tolerant(model, Path(dense_checkpoint), device)

    _expand_position_embeddings(model, max_seq_len)
    routing_info = apply_routing_variant_settings(model, var_config, attn_type, max_seq_len)
    routing_info.update(init_info)
    model._exp7_routing_info = routing_info  # type: ignore[attr-defined]
    return model, var_config


def _build_phase_aligned_samples(
    bench_cfg: LongContextBenchmarkConfig,
    train_context_length: int,
    n_samples: int,
    step: int,
) -> list[LongContextSample]:
    """Fresh procedural samples matching the active train curriculum settings."""
    gen = LongContextSampleGenerator(bench_cfg)
    depths = list(bench_cfg.needle_depths)
    tasks = list(bench_cfg.task_types)
    modes = list(bench_cfg.haystack_modes)
    rng = random.Random(int(bench_cfg.seed) + 900_000 + int(step))
    samples: list[LongContextSample] = []
    for i in range(n_samples):
        samples.append(
            gen.generate_one(
                context_length=train_context_length,
                needle_depth=depths[i % len(depths)],
                task_type=tasks[i % len(tasks)],
                haystack_mode=modes[i % len(modes)],
                seed=rng.randint(0, 2**31 - 1),
            )
        )
    return samples


@torch.no_grad()
def _eval_holdout(
    model: torch.nn.Module,
    bench_cfg: LongContextBenchmarkConfig,
    holdout_samples: list[LongContextSample],
    device: torch.device,
    *,
    show_progress: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout_samples)
    summary = evaluator.evaluate_module(
        model,
        device=device,
        max_samples=max_samples,
        show_progress=show_progress,
    )
    return {
        "overall_accuracy": summary.overall_accuracy,
        "primary_gate_accuracy": summary.primary_gate_accuracy,
        "primary_gate_correct": summary.primary_gate_correct,
        "primary_gate_total": summary.primary_gate_total,
        "secondary_accuracy": summary.secondary_accuracy,
        "secondary_correct": summary.secondary_correct,
        "secondary_total": summary.secondary_total,
        "pure_niah_accuracy": summary.primary_gate_accuracy,
        "pure_niah_correct": summary.primary_gate_correct,
        "pure_niah_total": summary.primary_gate_total,
        "correct": summary.correct,
        "total": summary.total,
        "errors": len(summary.errors),
        "by_task_type": dict(summary.by_task_type),
        "by_needle_depth": dict(summary.by_needle_depth),
    }


def _train_on_benchmark(
    model: torch.nn.Module,
    config: dict,
    bench_cfg: LongContextBenchmarkConfig,
    holdout_samples: list[LongContextSample],
    device: torch.device,
    train_context_length: int,
    logger,
    *,
    max_steps: int | None = None,
    training_stage: str = "train",
) -> dict[str, Any]:
    train_cfg = config.get("transformer", {})
    if max_steps is None:
        max_steps = _resolve_train_steps(config, training_stage)
    if max_steps <= 0:
        return {
            "trained_steps": 0,
            "train_context_length": train_context_length,
            "training_stage": training_stage,
        }

    _apply_scatter_curriculum_spec(config, train_context_length)
    lc_bench = config.get("long_context_benchmark", {})
    expanded_scatter = lc_bench.get("needle_scatter_curriculum")
    if expanded_scatter:
        bench_cfg.needle_scatter_curriculum = list(expanded_scatter)
        scatter_init = _curriculum_needle_scatter_settings(0, bench_cfg.needle_scatter_curriculum)
        for key, value in scatter_init.items():
            if key in _NEEDLE_SCATTER_CURRICULUM_KEYS:
                setattr(bench_cfg, key, value)
        logger.info(
            "Scatter curriculum: %d tiers; step=0 settings=%s",
            len(expanded_scatter),
            scatter_init,
        )

    log_every = int(train_cfg.get("log_every", 500))
    validate_every = int(train_cfg.get("validate_every", 0))
    validate_every_min = int(train_cfg.get("validate_every_min", 5000))
    # Mid-train holdout: suite defaults to >=5000 steps between checks; calibration lowers the floor.
    if 0 < validate_every < validate_every_min:
        validate_every = validate_every_min
    cal_cfg = config.get("dense_calibration", {})
    early_stop = bool(cal_cfg.get("early_stop", False))
    patience_checks = int(cal_cfg.get("patience_checks", 3))
    min_delta_pp = float(cal_cfg.get("min_delta_pp", 0.005))
    target_accuracy = cal_cfg.get("target_accuracy")
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)
    live_metrics = bool(cal_cfg.get("live_metrics", False))
    lr = float(train_cfg.get("lr", 3e-4))
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    data_cfg = config.get("data", {})
    attn_type = config.get("model", {}).get("attention_type", "")
    batch_size = _effective_train_batch_size(
        int(data_cfg.get("batch_size", 1)),
        train_context_length,
        attn_type,
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda" and bool(data_cfg.get("pin_memory", True))
    prefetch_batches = int(data_cfg.get("prefetch_batches", 2))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
    use_amp = bool(config.get("training", {}).get("use_amp", True)) and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )
    loader = get_long_context_dataloader(
        bench_cfg,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_batches=prefetch_batches if num_workers == 0 else 0,
        prefetch_factor=prefetch_factor,
        train_context_length=train_context_length,
    )
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    if batch_size != int(data_cfg.get("batch_size", 1)):
        logger.info(
            "T=%d attn=%s: batch_size capped %d -> %d for VRAM",
            train_context_length,
            attn_type,
            int(data_cfg.get("batch_size", 1)),
            batch_size,
        )
    # Pre-compile Triton / Flex kernels so step 0 is not a compile outlier.
    head_dim = model.d_model // model.n_heads
    if attn_type == "linear":
        kernel = warmup_fla_linear_kernels(
            device=device,
            n_heads=model.n_heads,
            head_dim=head_dim,
            context_length=train_context_length,
        )
        logger.info("FLA linear kernel warmup done (kernel=%s T=%d)", kernel, train_context_length)
    elif attn_type == "local":
        window = int(
            config.get("routing_attention", {}).get("local_window")
            or config.get("router", {}).get("local_window", 64)
        )
        warmup_flex_sliding_window(
            device=device,
            n_heads=model.n_heads,
            head_dim=head_dim,
            context_length=train_context_length,
            window_size=window,
        )
        logger.info("Flex sliding-window warmup done (window=%d T=%d)", window, train_context_length)
    elif attn_type in ("routing", "key_vector", "learned_address", "dense", "dense_flash"):
        # CUDA graph / Triton warmup for sparse or SDPA paths at train T.
        warm_hi = int(getattr(model, "token_emb", None).num_embeddings if hasattr(model, "token_emb") else 256)
        warm_ids = torch.randint(
            1, max(2, warm_hi), (1, train_context_length), device=device, dtype=torch.long
        )
        with amp_ctx:
            warm_out = model(input_ids=warm_ids)
            warm_logits = warm_out["logits"]
            if any(p.requires_grad for p in model.parameters()):
                warm_logits.sum().backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("Attention forward warmup done (attn=%s T=%d)", attn_type, train_context_length)

    model.train()
    losses: list[float] = []
    curriculum = list(getattr(bench_cfg, "context_curriculum", None) or [])
    suffix_curriculum = list(getattr(bench_cfg, "suffix_curriculum", None) or [])
    needle_scatter_curriculum = list(getattr(bench_cfg, "needle_scatter_curriculum", None) or [])
    synthetic_task_curriculum = list(getattr(bench_cfg, "synthetic_task_curriculum", None) or [])
    placement_episode_batches = int(getattr(bench_cfg, "placement_episode_batches", 0) or 0)
    if placement_episode_batches > 1:
        logger.info(
            "Placement episode training: %d batches per fixed needles+query (scatter varies)",
            placement_episode_batches,
        )
    active_train_t = train_context_length
    active_suffix_sig: tuple[tuple[str, Any], ...] | None = None
    active_scatter_sig: tuple[tuple[str, Any], ...] | None = None
    active_task_sig: tuple[tuple[str, Any], ...] | None = None
    data_iter = iter(loader)
    non_blocking = pin_memory and device.type == "cuda"
    mid_validations: list[dict[str, Any]] = []
    best_holdout_acc = -1.0
    best_holdout_step = 0
    best_state_dict: dict[str, Any] | None = None
    restore_best = bool(cal_cfg.get("restore_best_checkpoint", True))
    stale_checks = 0
    stopped_early = False
    loss_ema: float | None = None
    train_acc_ema: float | None = None
    ema_alpha = 0.05
    train_evaluator = LongContextEvaluator(bench_cfg)

    pbar = tqdm(
        range(max_steps),
        desc=f"train T={train_context_length}",
        leave=live_metrics,
    )
    for step in pbar:
        if curriculum:
            new_t = _curriculum_context_length(step, train_context_length, curriculum)
            if new_t != active_train_t:
                active_train_t = new_t
                loader = get_long_context_dataloader(
                    bench_cfg,
                    split="train",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_batches=prefetch_batches if num_workers == 0 else 0,
                    prefetch_factor=prefetch_factor,
                    train_context_length=active_train_t,
                )
                data_iter = iter(loader)
                logger.info("Context curriculum: T=%d at step=%d", active_train_t, step)

        if suffix_curriculum:
            suffix_settings = _curriculum_suffix_settings(step, suffix_curriculum)
            suffix_sig = tuple(sorted(suffix_settings.items()))
            if suffix_sig != active_suffix_sig:
                active_suffix_sig = suffix_sig
                for key, value in suffix_settings.items():
                    if key in _SUFFIX_CURRICULUM_KEYS:
                        setattr(bench_cfg, key, value)
                loader = get_long_context_dataloader(
                    bench_cfg,
                    split="train",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_batches=prefetch_batches if num_workers == 0 else 0,
                    prefetch_factor=prefetch_factor,
                    train_context_length=active_train_t,
                )
                data_iter = iter(loader)
                logger.info("Suffix curriculum at step=%d: %s", step, suffix_settings)

        if needle_scatter_curriculum:
            scatter_settings = _curriculum_needle_scatter_settings(step, needle_scatter_curriculum)
            scatter_sig = tuple(sorted(scatter_settings.items()))
            if scatter_sig != active_scatter_sig:
                active_scatter_sig = scatter_sig
                for key, value in scatter_settings.items():
                    if key in _NEEDLE_SCATTER_CURRICULUM_KEYS:
                        setattr(bench_cfg, key, value)
                loader = get_long_context_dataloader(
                    bench_cfg,
                    split="train",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_batches=prefetch_batches if num_workers == 0 else 0,
                    prefetch_factor=prefetch_factor,
                    train_context_length=active_train_t,
                )
                data_iter = iter(loader)
                logger.info("Needle scatter curriculum at step=%d: %s", step, scatter_settings)
                if live_metrics:
                    scatter_max = scatter_settings.get("scatter_placement_max")
                    scatter_label = "full" if scatter_max is None else str(scatter_max)
                    tqdm.write(
                        f"[scatter] step {step + 1:>6}  placement=0-{scatter_label}"
                    )

        if synthetic_task_curriculum:
            task_settings = _curriculum_synthetic_task_settings(step, synthetic_task_curriculum)
            task_sig = tuple(sorted(task_settings.items()))
            if task_sig != active_task_sig:
                active_task_sig = task_sig
                hop_count = task_settings.get("synthetic_hop_count")
                if hop_count is not None:
                    _apply_synthetic_hop_settings(bench_cfg, int(hop_count))
                for key, value in task_settings.items():
                    if key in _SYNTHETIC_TASK_CURRICULUM_KEYS and key != "synthetic_hop_count":
                        setattr(bench_cfg, key, value)
                decoys = task_settings.get("num_distractors")
                if decoys is not None and "synthetic_decoy_addrs" not in task_settings:
                    bench_cfg.synthetic_decoy_addrs = int(decoys)
                loader = get_long_context_dataloader(
                    bench_cfg,
                    split="train",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_batches=prefetch_batches if num_workers == 0 else 0,
                    prefetch_factor=prefetch_factor,
                    train_context_length=active_train_t,
                )
                data_iter = iter(loader)
                logger.info("Synthetic task curriculum at step=%d: %s", step, task_settings)

        batch = transfer_batch_to_device(
            next(data_iter), device, non_blocking=non_blocking, pin_memory=pin_memory
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            if _uses_pointer_output_head(model):
                ptr_batch = _pointer_batch_tensors(batch.get("meta"), device, model)
                if ptr_batch is None:
                    raise RuntimeError(
                        "pointer output head requires batch meta with label_mode='pointer_index'"
                    )
                question_index, pointer_target = ptr_batch
                forward_kwargs: dict[str, Any] = {
                    "input_ids": batch["input_ids"],
                    "attn_mask": batch.get("attention_mask"),
                    "question_index": question_index,
                }
                if getattr(model, "output_head", "") == "pointer_mlp" and getattr(
                    model, "pointer_target_mode", ""
                ) == "value_slots":
                    forward_kwargs["pointer_target_slot"] = pointer_target
                else:
                    forward_kwargs["pointer_target_index"] = pointer_target
                out = model(**forward_kwargs)
                loss = out["loss"]
                logits = out["pointer_logits"]
            elif _uses_pool_mlp_token_head(model):
                meta_list = batch.get("meta")
                if not meta_list or not _batch_uses_query_only_answer(meta_list):
                    raise RuntimeError(
                        "pool_mlp_token requires batch meta with label_mode='query_only_answer'"
                    )
                question_index = _question_index_batch_tensor(meta_list, device)
                out = model(
                    input_ids=batch["input_ids"],
                    attn_mask=batch.get("attention_mask"),
                    question_index=question_index,
                )
                token_logits = out["token_logits"]
                targets = _query_only_answer_target_tensor(meta_list, device)
                loss = torch.nn.functional.cross_entropy(token_logits, targets)
                logits = token_logits
            else:
                out = model(input_ids=batch["input_ids"], attn_mask=batch.get("attention_mask"))
                logits = out["logits"]
                meta_list = batch.get("meta")
                if _batch_uses_query_only_answer(meta_list):
                    loss = _query_only_answer_aligned_loss(
                        logits,
                        batch["labels"],
                        batch.get("loss_weights"),
                        meta_list,
                    )
                else:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch["labels"][:, 1:].contiguous()
                    shift_weights = batch.get("loss_weights")
                    if shift_weights is not None:
                        shift_weights = shift_weights[:, 1:].contiguous()
                    loss = _weighted_long_context_loss(shift_logits, shift_labels, shift_weights)
        with torch.no_grad():
            train_acc, train_correct, train_total = _train_batch_gate_accuracy(
                logits.detach(),
                batch.get("meta"),
                train_evaluator,
                bench_cfg,
            )
        loss.backward()
        at_step = step + 1
        if lr_warmup_steps > 0:
            warmup_scale = min(1.0, float(at_step) / float(lr_warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr * warmup_scale
        optimizer.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        loss_ema = loss_val if loss_ema is None else ema_alpha * loss_val + (1.0 - ema_alpha) * loss_ema
        train_acc_ema = (
            train_acc
            if train_acc_ema is None
            else ema_alpha * train_acc + (1.0 - ema_alpha) * train_acc_ema
        )
        if live_metrics:
            pbar.set_postfix(
                loss=f"{loss_val:.3f}",
                ema=f"{loss_ema:.3f}",
                acc=f"{train_acc * 100:.1f}%",
                acc_ema=f"{train_acc_ema * 100:.1f}%",
                best=f"{max(best_holdout_acc, 0.0) * 100:.1f}%",
                refresh=False,
            )
            logger.info(
                "[train] T=%d step=%d/%d loss=%.4f ema=%.4f train_acc=%.2f%% (%d/%d) "
                "train_acc_ema=%.2f%% lr=%.2e best_val=%.2f%%",
                train_context_length,
                at_step,
                max_steps,
                loss_val,
                loss_ema,
                train_acc * 100.0,
                train_correct,
                train_total,
                train_acc_ema * 100.0,
                lr,
                max(best_holdout_acc, 0.0) * 100.0,
            )
            if at_step == 1 or log_every <= 0 or at_step % log_every == 0:
                tqdm.write(
                    f"[train] step {at_step:>6}/{max_steps}  loss={loss_val:.4f}  ema={loss_ema:.4f}  "
                    f"train_acc={train_acc * 100:.2f}% ({train_correct}/{train_total})  "
                    f"train_acc_ema={train_acc_ema * 100:.2f}%  "
                    f"best_val={max(best_holdout_acc, 0.0) * 100:.2f}%"
                )
        elif log_every > 0 and step % log_every == 0:
            logger.info(
                "T=%d train step=%d loss=%.4f train_acc=%.2f%% (%d/%d)",
                train_context_length,
                step,
                loss_val,
                train_acc * 100.0,
                train_correct,
                train_total,
            )

        run_holdout = validate_every > 0 and (
            at_step % validate_every == 0 or at_step == max_steps
        )
        if run_holdout:
            model.eval()
            t_val = time.perf_counter()
            val = _eval_holdout(
                model,
                bench_cfg,
                holdout_samples,
                device,
                show_progress=live_metrics,
            )
            val_sec = time.perf_counter() - t_val
            val["eval_wall_sec"] = val_sec
            val["eval_subset"] = "mid_train"
            val["eval_samples"] = val["total"]
            phase_val: dict[str, Any] | None = None
            if bool(cal_cfg.get("phase_aligned_eval", False)):
                n_phase = int(cal_cfg.get("phase_aligned_samples", 32))
                phase_samples = _build_phase_aligned_samples(
                    bench_cfg,
                    train_context_length,
                    n_phase,
                    at_step,
                )
                t_phase = time.perf_counter()
                phase_val = _eval_holdout(
                    model,
                    bench_cfg,
                    phase_samples,
                    device,
                    show_progress=False,
                )
                phase_val["eval_wall_sec"] = time.perf_counter() - t_phase
                phase_val["eval_subset"] = "phase_aligned"
                phase_val["eval_samples"] = n_phase
                phase_val["scatter_placement_max"] = bench_cfg.scatter_placement_max
                phase_val["synthetic_hop_count"] = bench_cfg.synthetic_hop_count
                phase_val["num_distractors"] = bench_cfg.num_distractors
                val["phase_aligned"] = phase_val
            mid_validations.append({"step": at_step, **val})
            acc = float(val.get("primary_gate_accuracy", val.get("pure_niah_accuracy", val["overall_accuracy"])))
            if live_metrics:
                tqdm.write(
                    f"[val]  step {at_step:>6}  gate={acc * 100:.2f}%  ({val.get('primary_gate_correct', val['correct'])}/"
                    f"{val.get('primary_gate_total', val['total'])})  "
                    f"best={max(best_holdout_acc, acc) * 100:.2f}%  stale={stale_checks}  "
                    f"wall={val_sec:.1f}s"
                )
                for task, task_acc in sorted(val.get("by_task_type", {}).items()):
                    tqdm.write(f"       {task:<22} {float(task_acc) * 100:5.1f}%")
                if phase_val is not None:
                    phase_acc = float(
                        phase_val.get(
                            "primary_gate_accuracy",
                            phase_val.get("pure_niah_accuracy", phase_val["overall_accuracy"]),
                        )
                    )
                    scatter_max = bench_cfg.scatter_placement_max
                    scatter_label = "full" if scatter_max is None else str(scatter_max)
                    tqdm.write(
                        f"[phase] step {at_step:>6}  scatter=0-{scatter_label}  "
                        f"hops={bench_cfg.synthetic_hop_count}  dist={bench_cfg.num_distractors}  "
                        f"gate={phase_acc * 100:.2f}%  ({phase_val.get('primary_gate_correct', phase_val['correct'])}/"
                        f"{phase_val.get('primary_gate_total', phase_val['total'])})  "
                        f"wall={phase_val.get('eval_wall_sec', 0.0):.1f}s"
                    )
                logger.info(
                    "[val] T=%d step=%d acc=%.4f (%d/%d) best=%.2f%%@step%d stale=%d wall=%.1fs",
                    train_context_length,
                    at_step,
                    acc,
                    val["correct"],
                    val["total"],
                    max(best_holdout_acc, acc) * 100.0,
                    best_holdout_step,
                    stale_checks,
                    val_sec,
                )
            else:
                logger.info(
                    "T=%d holdout step=%d accuracy=%.4f (%d/%d)",
                    train_context_length,
                    at_step,
                    acc,
                    val["correct"],
                    val["total"],
                )
            model.train()
            if acc > best_holdout_acc + min_delta_pp:
                best_holdout_acc = acc
                best_holdout_step = at_step
                stale_checks = 0
                if restore_best:
                    best_state_dict = deepcopy(model.state_dict())
            elif acc <= best_holdout_acc + min_delta_pp:
                stale_checks += 1
            if early_stop and at_step < max_steps:
                if target_accuracy is not None and acc >= target_accuracy and stale_checks >= patience_checks:
                    stopped_early = True
                    logger.info(
                        "Early stop at step=%d (target_accuracy=%.4f, stale_checks=%d)",
                        at_step,
                        target_accuracy,
                        stale_checks,
                    )
                    break
                if stale_checks >= patience_checks:
                    stopped_early = True
                    logger.info(
                        "Early stop at step=%d (no holdout gain >%.4f for %d checks)",
                        at_step,
                        min_delta_pp,
                        patience_checks,
                    )
                    break

    trained_steps = (step + 1) if losses else 0
    if stopped_early:
        trained_steps = step + 1

    restored_best = False
    if restore_best and best_state_dict is not None and best_holdout_step > 0:
        model.load_state_dict(best_state_dict)
        restored_best = True
        logger.info(
            "Restored best holdout weights from step=%d (gate=%.2f%%)",
            best_holdout_step,
            best_holdout_acc * 100.0,
        )

    return {
        "trained_steps": trained_steps,
        "final_loss": losses[-1] if losses else None,
        "mid_validations": mid_validations,
        "best_holdout": {
            "step": best_holdout_step,
            "overall_accuracy": best_holdout_acc if best_holdout_acc >= 0 else None,
            "primary_gate_accuracy": best_holdout_acc if best_holdout_acc >= 0 else None,
        },
        "restored_best_checkpoint": restored_best,
        "early_stopped": stopped_early,
        "training_mode": "fixed_context_length",
        "training_stage": training_stage,
        "train_context_length": train_context_length,
        "train_seed": bench_cfg.seed,
        "holdout_seed": bench_cfg.holdout_seed,
        "use_amp": use_amp,
        "validate_every": validate_every,
        "fair_finetune": bool(config.get("routing_attention", {}).get("fair_finetune", True)),
    }


def run(
    variant: str | None = None,
    dry_run: bool = False,
    config_override: dict | None = None,
    variants: list[str] | None = None,
    skip_training: bool = False,
    train_context_length: int | None = None,
    run_mode: str = "full",
    training_stage: str | None = None,
    dense_checkpoint_path: str | Path | None = None,
    save_dense_checkpoint: str | Path | None = None,
    router_index_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(7, variant=variant, config_override=config_override)
    device = init_experiment_runtime(config)

    runner = ExperimentRunner(
        experiment_name=config["experiment"]["name"],
        config=config,
        dry_run=dry_run,
    )
    config = runner.config
    logger = setup_logging(runner.run_dir)
    metrics_logger = MetricsLogger(runner.tensorboard_dir, runner.stats_dir)
    logger.info("Starting Experiment 7: %s", config["experiment"]["description"])
    profile_meta = config.get("suite_active_profile", {})
    if profile_meta:
        logger.info(
            "Suite profile: %s (%s)",
            profile_meta.get("name", "?"),
            profile_meta.get("description", ""),
        )
    logger.info(
        "Model trunk: n_layers=%s d_model=%s",
        config.get("model", {}).get("n_layers"),
        config.get("model", {}).get("d_model"),
    )

    if train_context_length is None:
        raise ValueError(
            "train_context_length is required. The suite trains one fixed T per sub-experiment."
        )

    _apply_scatter_curriculum_spec(config, train_context_length)
    bench_cfg = _benchmark_config_from_run(config)
    if bench_cfg.benchmark_family == "synthetic":
        from routing_attention.benchmarks.long_context.config import apply_synthetic_family_profile

        bench_cfg = apply_synthetic_family_profile(bench_cfg)
    if dry_run:
        bench_cfg = bench_cfg.apply_dry_run_profile()
        if not skip_training:
            config.setdefault("transformer", {})["max_steps"] = 20
        config.setdefault("data", {})["batch_size"] = 1
    if train_context_length not in bench_cfg.context_lengths:
        raise ValueError(
            f"train_context_length={train_context_length} not in "
            f"benchmark context_lengths={bench_cfg.context_lengths}"
        )

    assert_expected_device(config, device)
    runtime_info = collect_device_info(device)
    backends = backend_status()
    logger.info("Runtime device: %s", runtime_info)
    logger.info("Attention backends: %s", backends)
    if not backends.get("fla_linear"):
        raise RuntimeError("flash-linear-attention (fla) required for linear baseline")
    if not backends.get("flex_sliding_window"):
        raise RuntimeError("PyTorch Flex Attention required for local window baseline")
    assert_production_backends_available()

    logger.info(
        "Sub-experiment T=%d | train_seed=%s holdout_seed=%s",
        train_context_length,
        bench_cfg.seed,
        bench_cfg.holdout_seed,
    )

    holdout_mid, holdout_full, holdout_meta = resolve_holdout_splits(
        config,
        bench_cfg,
        train_context_length,
    )
    if not holdout_full:
        raise RuntimeError(f"No holdout samples for context_length={train_context_length}")
    logger.info(
        "Held-out eval at T=%d: official=%d mid_train=%d (fixed grid, shared across variants)",
        train_context_length,
        holdout_meta["holdout_full_samples"],
        holdout_meta["holdout_mid_samples"],
    )

    variant_list = variants or list(config.get("variants", {}).keys())
    if variant:
        variant_list = [variant]

    two_stage = _is_two_stage(config)
    max_seq = train_context_length
    shared_base = _create_shared_base_model(config, max_seq) if not two_stage else None
    if shared_base is not None:
        logger.info("Shared dense backbone init enabled (seed=%s)", config.get("seed", 42))
    elif two_stage:
        logger.info("Two-stage protocol: %s", _training_protocol(config))

    dense_ckpt: Path | None = Path(dense_checkpoint_path) if dense_checkpoint_path else None
    router_idx_ckpt: Path | None = (
        Path(router_index_checkpoint_path) if router_index_checkpoint_path else None
    )

    results: dict[str, Any] = {
        "variants": {},
        "train_context_length": train_context_length,
        "benchmark_config": bench_cfg.to_dict(),
        "runtime": runtime_info,
        "attention_backends": backends,
        "dry_run": dry_run,
        "shared_init": shared_base is not None,
        "two_stage": two_stage,
        "dense_checkpoint_path": str(dense_ckpt) if dense_ckpt else None,
        "holdout_samples": holdout_meta["holdout_full_samples"],
        "holdout_mid_samples": holdout_meta["holdout_mid_samples"],
        "holdout": holdout_meta,
        "holdout_seed": bench_cfg.holdout_seed,
        "production_backends": production_manifest_for_variants(variant_list),
        "training_protocol": {
            "name": _training_protocol(config),
            "mode": "step_based_fixed_context_length",
            "run_mode": run_mode,
            "validate_every": int(config.get("transformer", {}).get("validate_every", 0)),
            "post_train_holdout_eval": run_mode == "full",
            "holdout_shared_across_variants": True,
            "n_layers": config.get("model", {}).get("n_layers"),
            "max_steps": int(config.get("transformer", {}).get("max_steps", 0)),
            "dense_pretrain_steps": _resolve_train_steps(config, "dense_pretrain"),
            "sparse_finetune_steps": _resolve_train_steps(config, "finetune_from_dense"),
            "fair_finetune": bool(config.get("routing_attention", {}).get("fair_finetune", True)),
        },
        "suite_profile": profile_meta.get("name"),
    }

    latency_warmup = int(config.get("evaluation", {}).get("benchmark_warmup", 2))
    latency_runs = int(config.get("evaluation", {}).get("benchmark_runs", 5))
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout_full)

    announce_step(
        f"1 — T={train_context_length}: train + eval attention variants",
        logger,
    )
    for var_name in variant_list:
        if training_stage:
            stage = training_stage
        elif run_mode == "latency_only":
            stage = "eval_only"
        elif skip_training:
            stage = "eval_only" if dense_ckpt else "skip_train"
        elif two_stage and var_name == "dense_flash":
            stage = "dense_pretrain"
        elif two_stage:
            stage = "finetune_from_dense"
        else:
            stage = "single_stage"

        if two_stage and stage == "finetune_from_dense":
            if dense_ckpt is None or not dense_ckpt.exists():
                raise FileNotFoundError(
                    f"Two-stage finetune for {var_name} at T={train_context_length} requires "
                    f"a dense checkpoint (missing: {dense_ckpt})"
                )

        logger.info(
            "=== T=%d Variant: %s (stage=%s) ===",
            train_context_length,
            var_name,
            stage,
        )
        reset_peak_vram(device)
        init_ckpt = dense_ckpt if stage in ("finetune_from_dense", "eval_only") else None
        model, var_config = _build_variant_model(
            config,
            var_name,
            device,
            max_seq,
            shared_base=shared_base,
            dense_checkpoint=init_ckpt,
            router_index_checkpoint=router_idx_ckpt,
        )
        model_info = verify_model_on_device(model, device)
        routing_info = getattr(model, "_exp7_routing_info", {})
        logger.info(
            "Variant %s on %s (%d params)",
            var_name,
            model_info["param_device"],
            model_info["n_params"],
        )
        if routing_info:
            logger.info("Routing setup: %s", routing_info)

        training_audit = _verify_staged_training_protocol(
            var_name,
            stage,
            routing_info,
            model,
            two_stage=two_stage,
        )
        logger.info("Staged-training audit: %s", training_audit)

        if run_mode == "latency_only":
            skip_train_this = True
            skip_eval_this = True
        elif stage == "eval_only":
            skip_train_this = True
            skip_eval_this = run_mode != "full"
        else:
            skip_train_this = skip_training or stage == "skip_train"
            skip_eval_this = False

        if not skip_train_this:
            train_info = _train_on_benchmark(
                model,
                var_config,
                bench_cfg,
                holdout_mid,
                device,
                train_context_length,
                logger,
                max_steps=_resolve_train_steps(config, stage),
                training_stage=stage,
            )
            if stage == "dense_pretrain" and save_dense_checkpoint:
                ckpt_path = _save_dense_checkpoint(
                    model,
                    Path(save_dense_checkpoint),
                    train_context_length=train_context_length,
                    trained_steps=train_info.get("trained_steps", 0),
                )
                train_info["dense_checkpoint_saved"] = ckpt_path
                logger.info("Saved dense checkpoint: %s", ckpt_path)
        else:
            train_info = {
                "trained_steps": 0,
                "train_context_length": train_context_length,
                "training_stage": stage,
                "skipped": run_mode == "latency_only" or stage == "eval_only",
                "dense_checkpoint": str(dense_ckpt) if dense_ckpt else None,
            }

        if skip_eval_this:
            from routing_attention.benchmarks.long_context.evaluation import EvalSummary

            summary = EvalSummary(
                overall_accuracy=0.0,
                total=0,
                correct=0,
            )
            peak_mb = peak_vram_mb(device)
        else:
            summary = evaluator.evaluate_module(model, device=device)
            peak_mb = peak_vram_mb(device)

        reset_peak_vram(device)
        latency = evaluator.benchmark_forward_latency(
            model,
            device=device,
            context_length=train_context_length,
            warmup=latency_warmup,
            runs=latency_runs,
        )
        peak_mb_latency = peak_vram_mb(device)
        if peak_mb_latency is not None:
            peak_mb = max(peak_mb or 0.0, peak_mb_latency)

        if summary.errors:
            logger.warning(
                "Variant %s had %d eval errors: %s",
                var_name,
                len(summary.errors),
                summary.errors[0].error[:200],
            )

        if not skip_eval_this:
            table = evaluator.format_tables(summary)
            print(table)
        elif run_mode == "latency_only":
            logger.info(
                "Variant %s: latency-only at T=%d (dense training skipped above profile cap)",
                var_name,
                train_context_length,
            )
        if latency.get("latency_ms") is not None:
            logger.info(
                "Latency T=%s: %.2f ms (%.0f tok/s)",
                train_context_length,
                latency["latency_ms"],
                latency.get("tokens_per_sec") or 0,
            )

        if not skip_eval_this:
            plots_dir = runner.plots_dir / f"T{train_context_length}" / var_name
            save_all_benchmark_plots(
                summary,
                [train_context_length],
                bench_cfg.needle_depths,
                plots_dir,
            )
            eval_path = evaluator.save_results(
                summary, runner.stats_dir / f"T{train_context_length}" / var_name
            )
        else:
            eval_path = None

        top_failures = [
            {
                "expected": r.expected,
                "predicted": r.predicted,
                "task_type": r.task_type,
                "context_length": r.context_length,
                "needle_depth": r.needle_depth,
            }
            for r in summary.records
            if not r.correct
        ][:20]

        results["variants"][var_name] = {
            "summary": summary.to_dict(),
            "train": train_info,
            "eval_path": str(eval_path),
            "model_device": model_info,
            "routing_setup": routing_info,
            "training_audit": training_audit,
            "peak_vram_mb": peak_mb,
            "eval_errors": len(summary.errors),
            "eval_latency_ms": latency.get("latency_ms"),
            "tokens_per_sec": latency.get("tokens_per_sec"),
            "latency_benchmark": latency,
            "top_failures": top_failures,
        }
        del model
        reset_peak_vram(device)
        metrics_logger.log_scalar(
            f"eval/T{train_context_length}/{var_name}/accuracy",
            summary.overall_accuracy,
            0,
        )

    if shared_base is not None:
        del shared_base
        reset_peak_vram(device)

    summary_path = runner.stats_dir / f"experiment_7_T{train_context_length}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Experiment 7 T=%d complete. summary=%s", train_context_length, summary_path)
    return results
