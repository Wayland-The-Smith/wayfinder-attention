"""Shared utilities for all experiment scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from routing_attention.data.datasets import get_dataloader, get_tokenizer
from routing_attention.data.loader import get_training_batch_iterator
from routing_attention.utils.cuda import configure_cuda_training
from routing_attention.models.learned_address import (
    PerLayerAddressBook,
    attach_address_book_to_model,
    build_address_book_from_config,
    ensure_address_book_on_model as _ensure_address_book_on_model,
)

ensure_address_book_on_model = _ensure_address_book_on_model
from routing_attention.models.router import (
    MultiScaleRouter,
    PerLayerRouter,
    RouterMLP,
    build_router_from_config,
)
# RouterMLP used in make_router_eval_fn isinstance check
from routing_attention.models.transformer import TransformerLM
from routing_attention.utils.config import (
    apply_variant,
    load_config,
    merge_configs,
    resolve_eval_every,
    resolve_validation_batches,
)
from routing_attention.utils.experiment import (
    find_transformer_checkpoint_in_run,
    resolve_checkpoint_path,
    resolve_run_dir,
)
from routing_attention.utils.checkpoint import load_checkpoint


def get_device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_recall_eval_limits(config: dict[str, Any], dry_run: bool = False) -> tuple[int, int]:
    """Return (max_eval_samples, max_eval_tokens) for final recall evaluation."""
    from routing_attention.evaluation.recall import resolve_max_eval_tokens

    eval_cfg = config.get("evaluation", {})
    seq_len = config.get("data", {}).get("seq_len", config.get("model", {}).get("max_seq_len", 512))
    max_samples = eval_cfg.get("max_eval_samples", 64)
    max_tokens = resolve_max_eval_tokens(
        seq_len,
        eval_cfg.get("max_eval_tokens", 0),
        dry_run=dry_run,
    )
    return max_samples, max_tokens


def get_benchmark_eval_config(config: dict[str, Any]) -> dict[str, int]:
    """Return benchmark timing knobs from evaluation config."""
    eval_cfg = config.get("evaluation", {})
    return {
        "num_runs": eval_cfg.get("benchmark_runs", 10),
        "num_warmup": eval_cfg.get("benchmark_warmup", 3),
    }


def announce_step(step: str, logger=None, detail: str = "") -> None:
    """Print and log the current experiment step before it runs."""
    line = "=" * 60
    msg = f">>> STEP: {step}"
    if detail:
        msg = f"{msg} | {detail}"
    print(f"\n{line}\n{msg}\n{line}", flush=True)
    if logger is not None:
        logger.info("STEP: %s%s", step, f" | {detail}" if detail else "")


def init_experiment_runtime(config: dict[str, Any]) -> torch.device:
    """Seed RNG, select device, and apply CUDA performance settings."""
    set_seed(config.get("seed", 42))
    device = get_device(config)
    configure_cuda_training(config)
    return device


def load_experiment_config(
    experiment_num: int,
    variant: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / f"experiment_{experiment_num}.yaml")
    config = merge_configs(base, exp_cfg)
    config = apply_variant(config, variant)
    if config_override:
        config = merge_configs(config, config_override)
    return config


def build_transformer(
    config: dict[str, Any],
    attention_type: str = "dense",
    router: nn.Module | None = None,
) -> TransformerLM:
    model_cfg = config["model"]
    data_cfg = config["data"]
    router_cfg = config.get("router", {})
    tokenizer = get_tokenizer(data_cfg["dataset"])
    cfg_vocab = model_cfg.get("vocab_size")
    if data_cfg.get("dataset") == "long_context" and cfg_vocab is not None:
        vocab_size = int(cfg_vocab)
    else:
        vocab_size = getattr(tokenizer, "vocab_size", cfg_vocab or 257)
    if data_cfg.get("dataset") == "mnist" and vocab_size < 257:
        raise ValueError(
            f"MNIST pixel tokens use ids 1..256 (pad=0); model.vocab_size must be >= 257, got {vocab_size}"
        )

    num_digit_classes = 0
    if data_cfg.get("dataset") == "mnist":
        num_digit_classes = int(model_cfg.get("num_digit_classes", 10))

    model_max_seq = int(model_cfg.get("max_seq_len", data_cfg["seq_len"]))
    retrieval_config = None
    if config:
        from routing_attention.retrieval.index import sync_retrieval_config_dict

        retrieval_config = sync_retrieval_config_dict(config.get("retrieval"), model_max_seq)

    return TransformerLM(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg.get("max_seq_len", data_cfg["seq_len"]),
        dropout=model_cfg["dropout"],
        attention_type=attention_type,
        router=router,
        routing_top_k=router_cfg.get("top_k", 32),
        local_window=router_cfg.get("local_window", 64),
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        num_digit_classes=num_digit_classes,
        output_head=model_cfg.get("output_head", "lm_token"),
        pointer_target_mode=model_cfg.get("pointer_target_mode", "full_sequence"),
        pointer_mlp_hidden=int(model_cfg.get("pointer_mlp_hidden", 2100)),
        num_pointer_slots=int(
            model_cfg.get(
                "num_pointer_slots",
                config.get("long_context_benchmark", {}).get("num_slot_quads", 50),
            )
        ),
        pool_mlp_positions=int(model_cfg.get("pool_mlp_positions", 16)),
        config=config if attention_type == "routing" and router is None else None,
        retrieval_config=retrieval_config,
    )


def build_router(config: dict[str, Any]) -> nn.Module:
    return build_router_from_config(config)


def build_address_book(config: dict[str, Any]) -> PerLayerAddressBook:
    return build_address_book_from_config(config)


def load_addresses_from_reuse(
    config: dict[str, Any],
    device: torch.device,
    model: nn.Module | None = None,
    artifact: str = "best.pt",
) -> tuple[PerLayerAddressBook, dict[str, Any] | None]:
    """Load per-layer address book from checkpoint; attach to model when provided."""
    address_book = build_address_book(config).to(device)
    ckpt_path = _resolve_reuse_ckpt(
        config, "address_checkpoint", default_artifact=artifact, subdir="addresses",
    )
    meta = None
    if ckpt_path and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, address_book, device=device, strict=False)
    if model is not None:
        attach_address_book_to_model(model, address_book)
    return address_book, meta


def _resolve_reuse_ckpt(
    config: dict[str, Any],
    key: str,
    default_artifact: str = "best.pt",
    subdir: str | None = None,
) -> Path | None:
    reuse_val = config.get("reuse", {}).get(key)
    if reuse_val and str(reuse_val).startswith("latest:"):
        exp = str(reuse_val).split(":", 1)[1]
        from routing_attention.utils.experiment import find_latest_run
        latest = find_latest_run(exp)
        if latest:
            parts = ["checkpoints"]
            if subdir:
                parts.append(subdir)
            parts.append(default_artifact)
            candidate = latest.joinpath(*parts)
            if candidate.exists():
                return candidate
    return resolve_checkpoint_path(reuse_val, artifact_name=default_artifact)


def load_transformer_from_reuse(
    config: dict[str, Any],
    device: torch.device,
    attention_type: str = "dense",
    router: nn.Module | None = None,
) -> tuple[TransformerLM, dict[str, Any] | None]:
    model = build_transformer(config, attention_type=attention_type, router=router).to(device)
    ckpt_path = resolve_reuse_transformer_checkpoint(config)
    meta = None
    if ckpt_path and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, model, device=device, strict=False)
    return model, meta


def resolve_routing_model_checkpoint(config: dict[str, Any]) -> Path | None:
    """Resolve fine-tuned routing LM checkpoint from Experiment 4."""
    reuse_val = config.get("reuse", {}).get("routing_model_checkpoint")
    if not reuse_val:
        return None
    if str(reuse_val).startswith("latest:"):
        exp = str(reuse_val).split(":", 1)[1]
        from routing_attention.utils.experiment import find_latest_run
        latest = find_latest_run(exp)
        if latest:
            for rel in ("routing_final.pt", "routing/routing_final.pt"):
                candidate = latest / "checkpoints" / rel
                if candidate.exists():
                    return candidate
        return None
    path = resolve_checkpoint_path(reuse_val)
    return path if path and path.exists() else None


def load_router_from_reuse(
    config: dict[str, Any],
    device: torch.device,
    artifact: str = "best.pt",
) -> tuple[nn.Module, dict[str, Any] | None]:
    router = build_router(config).to(device)
    ckpt_path = _resolve_reuse_ckpt(config, "router_checkpoint", default_artifact=artifact, subdir="router")
    meta = None
    if ckpt_path and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, router, device=device, strict=False)
    return router, meta


def _collection_config(config: dict[str, Any]) -> dict[str, Any]:
    """Use larger batch size for attention cache collection when configured."""
    coll_batch = config.get("data_collection", {}).get("batch_size")
    if not coll_batch:
        return config
    merged = {**config, "data": {**config["data"], "batch_size": coll_batch}}
    return merged


def resolve_cache_batch_counts(config: dict[str, Any]) -> tuple[int, int]:
    """Return (train_batches, holdout_batches) for router cache collection."""
    coll = config.get("data_collection", {})
    train_batches = coll.get("train_max_batches", coll.get("max_batches", 64))
    holdout_batches = coll.get("holdout_max_batches", 32)
    return int(train_batches), int(holdout_batches)


def resolve_router_training_layer_idx(config: dict[str, Any]) -> int:
    """Resolve absolute layer index for single-router training (mode != per_layer)."""
    layer_idx = int(config.get("data_collection", {}).get("layer_idx", -1))
    n_layers = int(config["model"]["n_layers"])
    return layer_idx if layer_idx >= 0 else n_layers + layer_idx


def router_cache_paths(runner, all_layers: bool = True) -> tuple[Path, Path]:
    """Return chunked cache directories (one batch file per disk write)."""
    if all_layers:
        train_path = runner.data_cache_dir / "attention_cache_train_all_layers"
        holdout_path = runner.data_cache_dir / "attention_cache_holdout_all_layers"
    else:
        train_path = runner.data_cache_dir / "attention_cache_train"
        holdout_path = runner.data_cache_dir / "attention_cache_holdout"
    return train_path, holdout_path


def get_collection_dataloader(
    config: dict[str, Any],
    split: str = "train",
):
    """Batch iterator for cache collection.

    MNIST uses the official train split for cache training and the held-out test
    split for evaluation caches. Legacy synthetic data reuses one RNG stream with
    a batch offset for holdout.
    """
    from routing_attention.data.loader import get_training_batch_iterator
    from routing_attention.data.mnist import get_mnist_batch_iterator, is_mnist_dataset

    cfg = _collection_config(config)
    device = get_device(cfg)
    data_cfg = cfg["data"]
    train_batches, holdout_batches = resolve_cache_batch_counts(cfg)

    if is_mnist_dataset(data_cfg["dataset"]):
        mnist_split = "test" if split == "holdout" else "train"
        max_batches = holdout_batches if split == "holdout" else train_batches
        return get_mnist_batch_iterator(
            cfg,
            device,
            split=mnist_split,
            infinite=False,
            max_batches=max_batches,
            start_batch=0,
        )

    if data_cfg.get("dataset") == "long_context":
        from copy import deepcopy
        from routing_attention.data.datasets import get_dataloader

        cfg = _collection_config(config)
        cfg.setdefault("data", {})["train_context_length"] = (
            cfg["data"].get("train_context_length") or cfg["data"].get("seq_len")
        )
        if split == "holdout":
            cfg = deepcopy(cfg)
            bench = cfg.get("long_context_benchmark", {})
            cfg["seed"] = int(bench.get("holdout_seed", cfg.get("seed", 42) + 999))
        return get_dataloader(
            dataset_name="long_context",
            seq_len=int(cfg["data"]["train_context_length"]),
            batch_size=int(cfg["data"].get("batch_size", 1)),
            num_workers=int(cfg["data"].get("num_workers", 0)),
            infinite=True,
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
            split="train",
            config=cfg,
        )

    skip_batches = train_batches if split == "holdout" else 0
    return get_training_batch_iterator(cfg, device, infinite=True, skip_batches=skip_batches)


def resolve_baseline_run_dir(config: dict[str, Any]) -> Path | None:
    """Resolve baseline run directory from reuse.baseline_run or attention cache path."""
    reuse = config.get("reuse", {})
    run_dir = resolve_run_dir(reuse.get("baseline_run"))
    if run_dir is not None:
        return run_dir
    for key in ("attention_cache_train", "attention_cache"):
        path = reuse.get(key)
        if path:
            resolved = resolve_checkpoint_path(str(path), artifact_name="manifest.json")
            if resolved is not None:
                return resolved.parent.parent if resolved.parent.name == "batches" else resolved.parent
    return None


def resolve_reuse_transformer_checkpoint(config: dict[str, Any]) -> Path | None:
    """Resolve transformer checkpoint from reuse config with baseline_run fallback."""
    ckpt = _resolve_reuse_ckpt(config, "transformer_checkpoint", subdir="transformer", default_artifact="final.pt")
    if ckpt is not None and ckpt.exists():
        return ckpt
    for artifact in ("final.pt", "step_010000.pt", "step_005000.pt", "best.pt"):
        ckpt = _resolve_reuse_ckpt(config, "transformer_checkpoint", subdir="transformer", default_artifact=artifact)
        if ckpt is not None and ckpt.exists():
            return ckpt
    run_dir = resolve_baseline_run_dir(config)
    if run_dir is not None:
        return find_transformer_checkpoint_in_run(run_dir)
    return None


def _cache_manifest_compatible(cache_dir: Path, per_head: bool, all_layers: bool) -> bool:
    from routing_attention.data.chunked_cache import is_chunked_cache

    if not is_chunked_cache(cache_dir):
        return False
    import json

    with open(cache_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    if bool(manifest.get("per_head", False)) != bool(per_head):
        return False
    if all_layers and not manifest.get("all_layers", False):
        return False
    return True


def resolve_reuse_attention_cache_paths(
    config: dict[str, Any],
    all_layers: bool = True,
    per_head: bool = False,
) -> tuple[Path | None, Path | None]:
    """Return (train_cache, holdout_cache) from reuse when compatible with this run."""
    reuse = config.get("reuse", {})
    run_dir = resolve_baseline_run_dir(config)

    train_src = reuse.get("attention_cache_train") or reuse.get("attention_cache")
    holdout_src = reuse.get("attention_cache_holdout")

    if run_dir is not None:
        data_cache = run_dir / "data_cache"
        if train_src is None:
            train_src = data_cache / (
                "attention_cache_train_all_layers" if all_layers else "attention_cache_train"
            )
        if holdout_src is None:
            holdout_src = data_cache / (
                "attention_cache_holdout_all_layers" if all_layers else "attention_cache_holdout"
            )

    if not train_src or not holdout_src:
        return None, None

    train_resolved = resolve_checkpoint_path(str(train_src), artifact_name="manifest.json")
    holdout_resolved = resolve_checkpoint_path(str(holdout_src), artifact_name="manifest.json")
    if train_resolved is None or holdout_resolved is None:
        return None, None

    train_dir = train_resolved.parent if train_resolved.name == "manifest.json" else train_resolved
    holdout_dir = holdout_resolved.parent if holdout_resolved.name == "manifest.json" else holdout_resolved
    if not _cache_manifest_compatible(train_dir, per_head, all_layers):
        return None, None
    if not _cache_manifest_compatible(holdout_dir, per_head, all_layers):
        return None, None
    return train_dir, holdout_dir


def write_idea_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Save human-readable description of what this run is testing."""
    path = run_dir / "idea_manifest.yaml"
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def collect_router_attention_caches(
    model: nn.Module,
    config: dict[str, Any],
    runner,
    device: torch.device,
    logger=None,
    all_layers: bool = True,
    per_head: bool = False,
    force_refresh: bool = False,
) -> tuple[Path, Path]:
    """Collect disjoint train and holdout attention caches from the same data distribution."""
    from routing_attention.data.mnist import is_mnist_dataset
    from routing_attention.training.trainer import collect_attention_dataset

    train_batches, holdout_batches = resolve_cache_batch_counts(config)
    train_path, holdout_path = router_cache_paths(runner, all_layers=all_layers)
    layer_idx = config["data_collection"].get("layer_idx", -1)

    from routing_attention.data.chunked_cache import cache_is_ready

    reuse_train, reuse_holdout = None, None
    if not force_refresh:
        reuse_train, reuse_holdout = resolve_reuse_attention_cache_paths(
            config, all_layers=all_layers, per_head=per_head,
        )
    if reuse_train is not None and reuse_holdout is not None:
        if logger:
            logger.info("Reusing TRAIN attention cache from %s (skip collection)", reuse_train)
            logger.info("Reusing HOLDOUT attention cache from %s (skip collection)", reuse_holdout)
        save_stats(runner.stats_dir, "cache_reuse", {
            "train_cache": str(reuse_train),
            "holdout_cache": str(reuse_holdout),
            "skipped_collection": True,
        })
        return reuse_train, reuse_holdout

    if not cache_is_ready(train_path):
        if logger:
            if is_mnist_dataset(config["data"]["dataset"]):
                logger.info(
                    "MNIST TRAIN cache: official 60k train split, %d batches (all_layers=%s)",
                    train_batches,
                    all_layers,
                )
            else:
                logger.info(
                    "Collecting TRAIN attention cache: %d batches (all_layers=%s)",
                    train_batches,
                    all_layers,
                )
        coll_cfg = config.get("data_collection", {})
        cache_dtype = torch.float16 if coll_cfg.get("cache_dtype", "float16") == "float16" else torch.float32
        collect_attention_dataset(
            model=model,
            dataloader=get_collection_dataloader(config, split="train"),
            device=device,
            layer_idx=layer_idx,
            max_batches=train_batches,
            output_path=train_path,
            per_head=per_head,
            all_layers=all_layers,
            split="train",
            batch_offset=0,
            cache_dtype=cache_dtype,
        )
    elif logger:
        logger.info("Using existing TRAIN cache at %s", train_path)

    if not cache_is_ready(holdout_path):
        if logger:
            if is_mnist_dataset(config["data"]["dataset"]):
                logger.info(
                    "MNIST HOLDOUT cache: official 10k test split, %d batches",
                    holdout_batches,
                )
            else:
                logger.info(
                    "Collecting HOLDOUT attention cache: %d batches (offset=%d)",
                    holdout_batches,
                    train_batches,
                )
        coll_cfg = config.get("data_collection", {})
        cache_dtype = torch.float16 if coll_cfg.get("cache_dtype", "float16") == "float16" else torch.float32
        collect_attention_dataset(
            model=model,
            dataloader=get_collection_dataloader(config, split="holdout"),
            device=device,
            layer_idx=layer_idx,
            max_batches=holdout_batches,
            output_path=holdout_path,
            per_head=per_head,
            all_layers=all_layers,
            split="holdout",
            batch_offset=train_batches,
            cache_dtype=cache_dtype,
        )
    elif logger:
        logger.info("Using existing HOLDOUT cache at %s", holdout_path)

    return train_path, holdout_path


def get_train_dataloader(config: dict[str, Any], infinite: bool = True, for_collection: bool = False):
    """
    Fastest batch source for the experiment configuration.

    Training (infinite): GPU-resident MNIST or synthetic generator on CUDA.
    Eval (finite): official MNIST test split or finite synthetic samples.
    """
    cfg = _collection_config(config) if for_collection else config
    device = get_device(cfg)
    if infinite:
        return get_training_batch_iterator(cfg, device, infinite=True, split="train")
    return get_eval_dataloader(cfg)


def get_eval_dataloader(config: dict[str, Any]):
    """Held-out evaluation batches (MNIST official test split)."""
    from routing_attention.data.mnist import is_mnist_dataset

    data_cfg = config["data"]
    eval_split = data_cfg.get("eval_split", "test")
    return get_dataloader(
        dataset_name=data_cfg["dataset"],
        seq_len=data_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers"),
        infinite=False,
        prefetch_factor=data_cfg.get("prefetch_factor", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        include_mask=True,
        split=eval_split if is_mnist_dataset(data_cfg["dataset"]) else "train",
        config=config,
    )


def make_transformer_eval_fn(config: dict[str, Any], device: torch.device):
    """Optional pixel-LM validation on the held-out MNIST test split."""
    from routing_attention.evaluation.metrics import evaluate_lm

    max_batches = resolve_validation_batches(config)

    def eval_fn(model):
        return evaluate_lm(
            model,
            get_eval_dataloader(config),
            device,
            max_batches=max_batches,
        )

    return eval_fn


def save_stats(stats_dir: Path, name: str, data: dict[str, Any]) -> Path:
    path = stats_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def resolve_attention_cache(config: dict[str, Any], runner, layer_idx: int | None = None) -> Path | None:
    """Resolve attention cache from reuse config or current run."""
    reuse_path = config.get("reuse", {}).get("attention_cache")
    if reuse_path:
        if str(reuse_path).startswith("latest:"):
            exp = str(reuse_path).split(":", 1)[1]
            from routing_attention.utils.experiment import find_latest_run
            latest = find_latest_run(exp)
            if latest:
                if layer_idx is not None:
                    candidate = latest / "data_cache" / f"layer_{layer_idx}_cache.pt"
                    if candidate.exists():
                        return candidate
                for name in (
                    "attention_cache_train_all_layers",
                    "attention_cache_holdout_all_layers",
                    "attention_cache_train_all_layers.pt",
                    "attention_cache_all_layers.pt",
                    "attention_cache_train.pt",
                    "attention_cache.pt",
                ):
                    candidate = latest / "data_cache" / name
                    if candidate.exists():
                        return candidate
        else:
            p = Path(reuse_path)
            if p.exists():
                return p

    if layer_idx is not None:
        local = runner.data_cache_dir / f"layer_{layer_idx}_cache.pt"
        if local.exists():
            return local
    for name in (
        "attention_cache_train_all_layers",
        "attention_cache_holdout_all_layers",
        "attention_cache_train_all_layers.pt",
        "attention_cache_all_layers.pt",
        "attention_cache_train.pt",
        "attention_cache.pt",
    ):
        local = runner.data_cache_dir / name
        if local.exists():
            return local
    return None


def make_router_eval_fn(
    top_k: int,
    layer_idx: int | None = None,
    eval_max_samples: int = 8,
):
    """Build eval callback using router retrieval_scores (subsamples cache when validating)."""
    from routing_attention.evaluation.recall import compute_recall_from_router

    def router_eval_fn(router, hidden, attention, li=None):
        router.eval()
        li = li if li is not None else layer_idx
        h, a = hidden, attention
        if eval_max_samples > 0 and h.shape[0] > eval_max_samples:
            idx = torch.randperm(h.shape[0], device=h.device)[:eval_max_samples]
            h, a = h[idx], a[idx]
        with torch.no_grad():
            if a.dim() == 4:
                a = a.mean(dim=1)
            if isinstance(router, (RouterMLP, PerLayerRouter, PerLayerAddressBook)):
                return compute_recall_from_router(h, a, router, layer_idx=li, k=top_k)
            return compute_recall_from_router(h, a, router, layer_idx=li, k=top_k)

    return router_eval_fn


def get_cache_for_layer(cache: dict, layer_idx: int) -> dict:
    """Extract single-layer cache from all_layers or direct cache."""
    if "layers" in cache:
        return cache["layers"][layer_idx]
    if cache.get("layer_idx") == layer_idx or layer_idx is None:
        return cache
    raise ValueError(f"Cache does not contain layer {layer_idx}")
