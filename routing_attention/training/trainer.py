"""Training loops for transformer and router models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from routing_attention.losses.attention_factorization import (
    kl_attention_loss,
    mse_attention_loss,
    multi_scale_routing_loss,
)
from routing_attention.losses.contrastive import batched_infonce_loss, sampled_infonce_loss
from routing_attention.models.learned_address import PerLayerAddressBook
from routing_attention.models.router import MultiScaleRouter, PerLayerRouter


def _is_per_layer_routing_container(module: nn.Module) -> bool:
    return isinstance(module, (PerLayerRouter, PerLayerAddressBook))
from routing_attention.utils.checkpoint import save_checkpoint
from routing_attention.utils.config import resolve_eval_every


def _should_validate(eval_every: int, step: int) -> bool:
    """Mid-training validation only when eval_every > 0 and step is a multiple."""
    return eval_every > 0 and step > 0 and step % eval_every == 0


def _random_lm_ce_baseline(model: nn.Module) -> float:
    """log(vocab_size) — cross-entropy for uniform next-token guessing."""
    vocab = getattr(getattr(model, "lm_head", None), "out_features", None)
    if vocab is None:
        vocab = getattr(getattr(model, "token_emb", None), "num_embeddings", 256)
    return math.log(max(int(vocab), 2))


def _random_digit_ce_baseline(model: nn.Module) -> float:
    """log(num_classes) — cross-entropy for uniform digit guessing."""
    n = getattr(model, "num_digit_classes", 0) or 0
    return math.log(max(int(n), 2))


def _to_device_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"]
    if input_ids.device != device:
        input_ids = input_ids.to(device, non_blocking=non_blocking)
    attn_mask = batch.get("attention_mask")
    if attn_mask is not None and attn_mask.device != device:
        attn_mask = attn_mask.to(device, non_blocking=non_blocking)
    labels = batch.get("labels")
    if labels is not None and labels.device != device:
        labels = labels.to(device, non_blocking=non_blocking)
    return input_ids, attn_mask, labels


@torch.inference_mode()
def _batch_routing_recall_vs_random(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    router: nn.Module,
    layer_idx: int | None,
    top_k: int,
) -> tuple[float, float, float]:
    """Mini-batch Recall@K vs random-unit-vector baseline (for per-step logging)."""
    from routing_attention.evaluation.recall import compute_recall_at_k, compute_recall_from_router

    if attention.dim() == 4:
        attention = attention.mean(dim=1)
    metrics = compute_recall_from_router(
        hidden, attention, router, layer_idx=layer_idx, k=top_k,
    )
    recall_key = f"recall@{min(top_k, hidden.shape[1])}"
    recall = float(metrics.get(recall_key, 0.0))
    rand = torch.randn(hidden.shape[0], hidden.shape[1], hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    rand = rand / rand.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    random_metrics = compute_recall_at_k(rand, attention, k=top_k)
    random_recall = float(random_metrics.get(recall_key, 0.0))
    return recall, random_recall, recall - random_recall


def _router_loss_tag(layer_idx: int | None, loss_type: str) -> str:
    if layer_idx is not None:
        return f"router/layer_{layer_idx}/{loss_type}_loss"
    return f"router/{loss_type}_loss"


def _to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    non_blocking = device.type == "cuda"
    input_ids = batch["input_ids"]
    if input_ids.device != device:
        input_ids = input_ids.to(device, non_blocking=non_blocking)
    attn_mask = batch.get("attention_mask")
    if attn_mask is not None and attn_mask.device != device:
        attn_mask = attn_mask.to(device, non_blocking=non_blocking)
    return input_ids, attn_mask


class TransformerTrainer:
    """Train a small transformer LM with optional routing auxiliary loss."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        runner,
        metrics_logger,
        logger,
        max_steps: int = 10000,
        eval_every: int = 500,
        save_every: int = 5000,
        eval_fn: Callable | None = None,
        freeze_router: bool = False,
        routing_aux_weight: float = 0.0,
        routing_aux_loss_type: str = "infonce",
        routing_top_k: int = 32,
        routing_temperature: float = 0.07,
        teacher_model: nn.Module | None = None,
        routing_aux_layer: int = -1,
        aux_hidden_source: str = "student",
        aux_attention_source: str = "teacher",
        digit_loss_weight: float = 0.0,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.runner = runner
        self.metrics_logger = metrics_logger
        self.logger = logger
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.save_every = save_every
        self.eval_fn = eval_fn
        self.freeze_router = freeze_router
        self.routing_aux_weight = routing_aux_weight
        self.routing_aux_loss_type = routing_aux_loss_type
        self.routing_top_k = routing_top_k
        self.routing_temperature = routing_temperature
        self.teacher_model = teacher_model
        self.routing_aux_layer = routing_aux_layer
        self.aux_hidden_source = aux_hidden_source
        self.aux_attention_source = aux_attention_source
        self.digit_loss_weight = digit_loss_weight
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.global_step = 0
        self.best_loss = float("inf")

        if freeze_router and hasattr(model, "freeze_router"):
            model.freeze_router()
            self.logger.info("Router frozen during LM training")

    def _compute_routing_aux_loss(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary routing loss using dense teacher attention on a single batch."""
        if self.teacher_model is None or self.routing_aux_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        layer = self.routing_aux_layer
        if layer < 0:
            layer = self.teacher_model.n_layers + layer

        # Hidden states: student (default) or teacher — see aux_hidden_source config
        if self.aux_hidden_source == "teacher":
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_out = self.teacher_model(
                    input_ids, attn_mask=attn_mask,
                    return_pre_attention_hidden=True, return_attentions=True,
                )
            hidden = teacher_out["pre_attention_hidden"][layer]
        else:
            student_out = self.model(
                input_ids, attn_mask=attn_mask,
                return_pre_attention_hidden=True, return_attentions=True,
            )
            hidden = student_out["pre_attention_hidden"][layer]

        # Attention targets: always from teacher dense model (ground truth neighborhoods)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids, attn_mask=attn_mask, return_attentions=True,
            )
        attention = teacher_out["attentions"][layer]
        if attention.dim() == 4:
            attention = attention.mean(dim=1)

        router = getattr(self.model, "router", None)
        if router is None:
            return torch.tensor(0.0, device=self.device)

        router_module = router.get_router(layer) if isinstance(router, PerLayerRouter) else router
        if isinstance(router, MultiScaleRouter):
            local_r, global_r = router.routing_for_loss(hidden)
            return multi_scale_routing_loss(
                local_r, global_r, attention,
                top_k=self.routing_top_k,
                temperature=self.routing_temperature,
                loss_type=self.routing_aux_loss_type,
            )

        return _routing_loss_from_hidden(
            hidden, attention, router_module, router, layer,
            self.routing_aux_loss_type, self.routing_top_k, self.routing_temperature,
        )

    def train(self, dataloader) -> dict[str, Any]:
        self.model.train()
        if self.freeze_router and hasattr(self.model, "freeze_router"):
            self.model.freeze_router()

        lm_baseline = _random_lm_ce_baseline(self.model)
        digit_baseline = _random_digit_ce_baseline(self.model)
        has_digit = getattr(self.model, "digit_head", None) is not None and self.digit_loss_weight > 0
        desc = f"Phase A | PixelLM+Digit (rand CE={lm_baseline:.3f}/{digit_baseline:.3f})" if has_digit else (
            f"Phase A | task=PixelLM (random CE={lm_baseline:.4f})"
        )
        pbar = tqdm(total=self.max_steps, desc=desc)
        data_iter = iter(dataloader)

        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids, attn_mask, labels = _to_device_batch(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(input_ids, attn_mask=attn_mask, labels=labels)
                lm_loss = out["loss"]
                total_loss = lm_loss
                digit_loss = out.get("digit_loss")
                if digit_loss is not None and self.digit_loss_weight > 0:
                    total_loss = total_loss + self.digit_loss_weight * digit_loss
                if self.routing_aux_weight > 0:
                    aux = self._compute_routing_aux_loss(input_ids, attn_mask)
                    total_loss = total_loss + self.routing_aux_weight * aux

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            lm_val = lm_loss.item()
            vs_random = lm_val - lm_baseline
            beats_random = vs_random < 0
            for tag in ("task/lm_loss", "phase_a/lm_loss", "transformer/lm_loss"):
                self.metrics_logger.log_scalar(tag, lm_val, self.global_step)
            for tag in ("task/lm_vs_random", "phase_a/lm_vs_random"):
                self.metrics_logger.log_scalar(tag, vs_random, self.global_step)
            self.metrics_logger.log_scalar(
                "phase_a/lm_beats_random", float(beats_random), self.global_step
            )
            postfix = {
                "lm_loss": f"{lm_val:.4f}",
                "vs_random": f"{vs_random:+.4f}",
                "beats_random": "yes" if beats_random else "no",
            }
            digit_loss_tensor = out.get("digit_loss")
            if digit_loss_tensor is not None:
                digit_val = digit_loss_tensor.item()
                digit_vs_random = digit_val - digit_baseline
                digit_beats = digit_vs_random < 0
                digit_acc = out.get("digit_accuracy")
                digit_acc_val = float(digit_acc.item()) if digit_acc is not None else 0.0
                for tag in ("task/digit_loss", "phase_a/digit_loss"):
                    self.metrics_logger.log_scalar(tag, digit_val, self.global_step)
                for tag in ("task/digit_vs_random", "phase_a/digit_vs_random"):
                    self.metrics_logger.log_scalar(tag, digit_vs_random, self.global_step)
                self.metrics_logger.log_scalar(
                    "phase_a/digit_beats_random", float(digit_beats), self.global_step
                )
                self.metrics_logger.log_scalar(
                    "phase_a/digit_accuracy", digit_acc_val, self.global_step
                )
                postfix.update(
                    digit_loss=f"{digit_val:.4f}",
                    digit_vs_random=f"{digit_vs_random:+.4f}",
                    digit_acc=f"{digit_acc_val:.3f}",
                )
            if self.routing_aux_weight > 0:
                aux_val = aux.item()
                for tag in ("task/routing_loss", "phase_a/routing_aux_loss", "router/aux_loss"):
                    self.metrics_logger.log_scalar(tag, aux_val, self.global_step)
                self.metrics_logger.log_scalar(
                    "transformer/total_loss", total_loss.item(), self.global_step
                )
            pbar.update(1)
            pbar.set_postfix(**postfix, refresh=False)
            if self.global_step == 1 or self.global_step % 1000 == 0:
                msg = (
                    f"Phase A step {self.global_step} | lm_loss={lm_val:.4f} | lm_vs_random={vs_random:+.4f}"
                )
                if digit_loss_tensor is not None:
                    msg += (
                        f" | digit_loss={digit_val:.4f} | digit_vs_random={digit_vs_random:+.4f} "
                        f"| digit_acc={digit_acc_val:.3f}"
                    )
                msg += " | routing_loss=N/A"
                self.logger.info(msg)
                self.metrics_logger.flush_history()

            if self.eval_fn and _should_validate(self.eval_every, self.global_step):
                eval_metrics = self.eval_fn(self.model)
                self.metrics_logger.log_dict(eval_metrics, self.global_step, prefix="eval")
                if eval_metrics.get("loss", float("inf")) < self.best_loss:
                    self.best_loss = eval_metrics["loss"]
                    save_checkpoint(
                        self.runner.checkpoint_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        step=self.global_step,
                        metrics=eval_metrics,
                    )

            if self.global_step % self.save_every == 0:
                save_checkpoint(
                    self.runner.checkpoint_dir / f"step_{self.global_step:06d}.pt",
                    self.model,
                    self.optimizer,
                    step=self.global_step,
                )

        pbar.close()
        save_checkpoint(
            self.runner.checkpoint_dir / "final.pt",
            self.model,
            self.optimizer,
            step=self.global_step,
        )
        return {"steps": self.global_step, "best_loss": self.best_loss}


class RouterTrainer:
    """Train routing module on frozen transformer hidden states + attention."""

    def __init__(
        self,
        router: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        runner,
        metrics_logger,
        logger,
        loss_type: str = "infonce",
        top_k: int = 32,
        temperature: float = 0.07,
        max_steps: int = 4000,
        eval_every: int = 200,
        save_every: int = 4000,
        eval_fn: Callable | None = None,
        batch_size: int = 4,
        num_negatives: int = 64,
        layer_idx: Optional[int] = None,
        attention_supervision: str = "head_avg",
        eval_max_samples: int = 8,
        use_amp: bool = True,
    ):
        self.router = router
        self.optimizer = optimizer
        self.device = device
        self.runner = runner
        self.metrics_logger = metrics_logger
        self.logger = logger
        self.loss_type = loss_type
        self.top_k = top_k
        self.temperature = temperature
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.save_every = save_every
        self.eval_fn = eval_fn
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.layer_idx = layer_idx
        self.attention_supervision = attention_supervision
        self.eval_max_samples = eval_max_samples
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.global_step = 0
        self.best_recall = 0.0

    def _compute_loss(
        self,
        hidden: torch.Tensor,
        attention: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        if isinstance(self.router, MultiScaleRouter):
            local_r, global_r = self.router.routing_for_loss(hidden)
            attn = attention.mean(dim=1) if attention.dim() == 4 else attention
            return multi_scale_routing_loss(
                local_r, global_r, attn,
                top_k=self.top_k, temperature=self.temperature,
                loss_type=self.loss_type if self.loss_type in ("infonce", "infonce_sampled") else "infonce",
            )

        router_module = self.router.get_router(layer_idx) if _is_per_layer_routing_container(self.router) else self.router

        # Per-head supervision: train against each head's attention pattern separately
        if attention.dim() == 4 and self.attention_supervision == "per_head":
            losses = []
            for h in range(attention.shape[1]):
                losses.append(_routing_loss_from_hidden(
                    hidden, attention[:, h], router_module, self.router, layer_idx,
                    self.loss_type, self.top_k, self.temperature, self.num_negatives,
                ))
            return torch.stack(losses).mean()

        if attention.dim() == 4:
            attention = attention.mean(dim=1)

        return _routing_loss_from_hidden(
            hidden, attention, router_module, self.router, layer_idx,
            self.loss_type, self.top_k, self.temperature, self.num_negatives,
        )

    def train_on_cache(
        self,
        cache_path: Path,
        holdout_cache_path: Path | None = None,
    ) -> dict[str, Any]:
        """Train router on pre-collected hidden states + attention (step-based, no epochs)."""
        from routing_attention.data.chunked_cache import is_chunked_cache

        if is_chunked_cache(cache_path):
            return self.train_on_chunked_cache(
                cache_path,
                holdout_cache_path=holdout_cache_path,
                layer_idx=self.layer_idx,
            )
        data = torch.load(cache_path, map_location=self.device, weights_only=False)
        eval_hidden, eval_attention = None, None
        if holdout_cache_path is not None and holdout_cache_path.exists():
            holdout = torch.load(holdout_cache_path, map_location=self.device, weights_only=False)
            eval_hidden = holdout["hidden_states"].contiguous()
            eval_attention = holdout["attention"].contiguous()
        return self.train_on_tensors(
            data["hidden_states"].contiguous(),
            data["attention"].contiguous(),
            layer_idx=data.get("layer_idx", self.layer_idx),
            eval_hidden=eval_hidden,
            eval_attention=eval_attention,
        )

    def train_on_chunked_cache(
        self,
        cache_dir: Path,
        holdout_cache_path: Path | None = None,
        layer_idx: int | None = None,
    ) -> dict[str, Any]:
        """Train router by sampling random on-disk batches (low RAM)."""
        import random as py_random

        from routing_attention.data.chunked_cache import ChunkedAttentionCache, is_chunked_cache

        from routing_attention.data.chunked_cache import normalize_layer_idx

        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        train_cache = ChunkedAttentionCache(cache_dir)
        if layer_idx is None:
            raise ValueError(
                "RouterTrainer.layer_idx is required for chunked cache training "
                "(set data_collection.layer_idx for router mode=single)."
            )
        layer_idx = normalize_layer_idx(layer_idx, train_cache.n_layers)
        holdout_cache = (
            ChunkedAttentionCache(holdout_cache_path)
            if holdout_cache_path is not None and is_chunked_cache(holdout_cache_path)
            else None
        )

        eval_hidden, eval_attention = None, None
        if holdout_cache is not None:
            eval_hidden, eval_attention = holdout_cache.load_layer_tensors(
                layer_idx,
                self.device,
                max_samples=self.eval_max_samples,
            )

        self.router.train()
        loss_tag = _router_loss_tag(layer_idx, self.loss_type)
        layer_label = f"L{layer_idx}" if layer_idx is not None else "all"
        pbar = tqdm(
            total=self.max_steps,
            desc=f"Phase C | task=Routing {layer_label} ({self.loss_type})",
        )
        rng = py_random.Random(self.global_step)

        for step in range(self.max_steps):
            h_batch, a_batch = train_cache.random_batch_layer(layer_idx, self.device, rng=rng)
            if h_batch.shape[0] > self.batch_size:
                idx = torch.randint(0, h_batch.shape[0], (self.batch_size,), device=self.device)
                h_batch = h_batch[idx]
                a_batch = a_batch[idx]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self._compute_loss(h_batch, a_batch, layer_idx=layer_idx)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step = step + 1
            router_val = loss.item()
            recall, random_recall, recall_vs_random = _batch_routing_recall_vs_random(
                h_batch, a_batch, self.router, layer_idx, self.top_k,
            )
            for tag in ("task/routing_loss", "phase_c/routing_loss"):
                self.metrics_logger.log_scalar(tag, router_val, self.global_step)
            self.metrics_logger.log_scalar(loss_tag, router_val, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_recall", recall, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_random_recall", random_recall, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_vs_random", recall_vs_random, self.global_step)
            self.metrics_logger.log_scalar(
                "phase_c/routing_beats_random", float(recall_vs_random > 0), self.global_step
            )
            pbar.update(1)
            pbar.set_postfix(
                routing_loss=f"{router_val:.4f}",
                recall=f"{recall:.3f}",
                vs_random=f"{recall_vs_random:+.3f}",
                refresh=False,
            )
            if self.global_step == 1 or self.global_step % 1000 == 0:
                self.logger.info(
                    "Phase C layer %s step %d | routing_loss=%.4f | recall=%.3f | "
                    "random_recall=%.3f | routing_vs_random=%+.3f",
                    layer_label,
                    self.global_step,
                    router_val,
                    recall,
                    random_recall,
                    recall_vs_random,
                )
                self.metrics_logger.flush_history()

            if self.eval_fn and eval_hidden is not None and _should_validate(self.eval_every, self.global_step):
                try:
                    eval_metrics = self.eval_fn(self.router, eval_hidden, eval_attention, layer_idx)
                except TypeError:
                    eval_metrics = self.eval_fn(self.router, eval_hidden, eval_attention)
                self.metrics_logger.log_dict(eval_metrics, self.global_step, prefix="eval/holdout")
                recall_key = f"recall@{min(self.top_k, eval_hidden.shape[1])}"
                holdout_recall = eval_metrics.get(recall_key, 0)
                if holdout_recall > self.best_recall:
                    self.best_recall = holdout_recall
                    save_checkpoint(
                        self.runner.checkpoint_dir / "best.pt",
                        self.router,
                        self.optimizer,
                        step=self.global_step,
                        metrics=eval_metrics,
                        extra={"loss_type": self.loss_type, "layer_idx": layer_idx},
                    )
                    self.logger.info(
                        "Phase C layer %s step %d | new holdout best %s=%.4f",
                        layer_label,
                        self.global_step,
                        recall_key,
                        holdout_recall,
                    )

            if self.global_step % self.save_every == 0:
                save_checkpoint(
                    self.runner.checkpoint_dir / f"step_{self.global_step:06d}.pt",
                    self.router,
                    self.optimizer,
                    step=self.global_step,
                )

        pbar.close()
        save_checkpoint(
            self.runner.checkpoint_dir / "final.pt",
            self.router,
            self.optimizer,
            step=self.global_step,
            extra={"loss_type": self.loss_type, "layer_idx": layer_idx},
        )
        return {
            "steps": self.global_step,
            "best_recall": self.best_recall,
            "loss_type": self.loss_type,
            "layer_idx": layer_idx,
        }

    def train_on_tensors(
        self,
        hidden: torch.Tensor,
        attention: torch.Tensor,
        layer_idx: int | None = None,
        eval_hidden: torch.Tensor | None = None,
        eval_attention: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Train router from in-memory tensors (avoids per-layer disk round-trips)."""
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        n_samples = hidden.shape[0]

        if eval_hidden is not None and eval_attention is not None:
            holdout_hidden, holdout_attention = eval_hidden, eval_attention
        else:
            holdout_hidden, holdout_attention = hidden, attention

        if self.eval_max_samples > 0 and holdout_hidden.shape[0] > self.eval_max_samples:
            eval_idx = torch.randperm(holdout_hidden.shape[0], device=self.device)[: self.eval_max_samples]
            eval_hidden, eval_attention = holdout_hidden[eval_idx], holdout_attention[eval_idx]
        else:
            eval_hidden, eval_attention = holdout_hidden, holdout_attention

        self.router.train()
        loss_tag = _router_loss_tag(layer_idx, self.loss_type)
        layer_label = f"L{layer_idx}" if layer_idx is not None else "all"
        pbar = tqdm(
            total=self.max_steps,
            desc=f"Phase C | task=Routing {layer_label} ({self.loss_type})",
        )

        for step in range(self.max_steps):
            if n_samples > self.batch_size:
                idx = torch.randint(0, n_samples, (self.batch_size,), device=self.device)
                h_batch = hidden[idx]
                a_batch = attention[idx]
            else:
                h_batch, a_batch = hidden, attention

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self._compute_loss(h_batch, a_batch, layer_idx=layer_idx)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step = step + 1
            router_val = loss.item()
            recall, random_recall, recall_vs_random = _batch_routing_recall_vs_random(
                h_batch, a_batch, self.router, layer_idx, self.top_k,
            )
            for tag in ("task/routing_loss", "phase_c/routing_loss"):
                self.metrics_logger.log_scalar(tag, router_val, self.global_step)
            self.metrics_logger.log_scalar(loss_tag, router_val, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_recall", recall, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_random_recall", random_recall, self.global_step)
            self.metrics_logger.log_scalar("phase_c/routing_vs_random", recall_vs_random, self.global_step)
            self.metrics_logger.log_scalar(
                "phase_c/routing_beats_random", float(recall_vs_random > 0), self.global_step
            )
            pbar.update(1)
            pbar.set_postfix(
                routing_loss=f"{router_val:.4f}",
                recall=f"{recall:.3f}",
                vs_random=f"{recall_vs_random:+.3f}",
                refresh=False,
            )
            if self.global_step == 1 or self.global_step % 1000 == 0:
                self.logger.info(
                    "Phase C layer %s step %d | routing_loss=%.4f | recall=%.3f | "
                    "random_recall=%.3f | routing_vs_random=%+.3f",
                    layer_label,
                    self.global_step,
                    router_val,
                    recall,
                    random_recall,
                    recall_vs_random,
                )
                self.metrics_logger.flush_history()

            if self.eval_fn and _should_validate(self.eval_every, self.global_step):
                try:
                    eval_metrics = self.eval_fn(self.router, eval_hidden, eval_attention, layer_idx)
                except TypeError:
                    eval_metrics = self.eval_fn(self.router, eval_hidden, eval_attention)
                self.metrics_logger.log_dict(eval_metrics, self.global_step, prefix="eval/holdout")
                recall_key = f"recall@{min(self.top_k, hidden.shape[1])}"
                holdout_recall = eval_metrics.get(recall_key, 0)
                if holdout_recall > self.best_recall:
                    self.best_recall = holdout_recall
                    save_checkpoint(
                        self.runner.checkpoint_dir / "best.pt",
                        self.router,
                        self.optimizer,
                        step=self.global_step,
                        metrics=eval_metrics,
                        extra={"loss_type": self.loss_type, "layer_idx": layer_idx},
                    )
                    self.logger.info(
                        "Phase C layer %s step %d | new holdout best %s=%.4f",
                        layer_label,
                        self.global_step,
                        recall_key,
                        holdout_recall,
                    )

            if self.global_step % self.save_every == 0:
                save_checkpoint(
                    self.runner.checkpoint_dir / f"step_{self.global_step:06d}.pt",
                    self.router,
                    self.optimizer,
                    step=self.global_step,
                )

        pbar.close()
        save_checkpoint(
            self.runner.checkpoint_dir / "final.pt",
            self.router,
            self.optimizer,
            step=self.global_step,
            extra={"loss_type": self.loss_type, "layer_idx": layer_idx},
        )
        return {
            "steps": self.global_step,
            "best_recall": self.best_recall,
            "loss_type": self.loss_type,
            "layer_idx": layer_idx,
        }

    # Backward-compatible alias (training is step-based, not epoch-based)
    train_epoch_on_cache = train_on_cache


def _routing_loss_from_hidden(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    router_module: nn.Module,
    router_container: nn.Module,
    layer_idx: Optional[int],
    loss_type: str,
    top_k: int,
    temperature: float,
    num_negatives: int = 64,
) -> torch.Tensor:
    """Differentiable routing loss — gradients train the routing MLP, not top-k selection."""
    if hasattr(router_module, "forward_query_key"):
        q, k = (
            router_container.get_router(layer_idx).forward_query_key(hidden)
            if _is_per_layer_routing_container(router_container) and layer_idx is not None
            else router_module.forward_query_key(hidden)
        )
    else:
        q = (
            router_container.get_router(layer_idx)(hidden)
            if _is_per_layer_routing_container(router_container) and layer_idx is not None
            else router_module(hidden)
        )
        k = q

    if loss_type == "infonce":
        return batched_infonce_loss(q, attention, key_routing=k, top_k=top_k, temperature=temperature)
    if loss_type == "infonce_sampled":
        return sampled_infonce_loss(
            q, attention, key_routing=k, top_k=top_k, temperature=temperature, num_negatives=num_negatives,
        )
    if loss_type == "mse":
        return mse_attention_loss(q, attention, key_routing=k, router=router_module)
    if loss_type == "kl":
        return kl_attention_loss(q, attention, key_routing=k, temperature=temperature, router=router_module)
    raise ValueError(f"Unknown loss_type: {loss_type}")


@torch.inference_mode()
def collect_attention_dataset(
    model: nn.Module,
    dataloader,
    device: torch.device,
    layer_idx: int = -1,
    max_batches: int = 50,
    output_path: Path | None = None,
    per_head: bool = False,
    all_layers: bool = False,
    split: str = "train",
    batch_offset: int = 0,
    cache_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """
    Collect hidden states and attention weights from frozen transformer.

    If all_layers=True, saves per-layer caches for per-layer router training.
    """
    model.eval()

    if all_layers:
        from routing_attention.data.chunked_cache import ChunkedCacheWriter

        n_layers = model.n_layers
        writer = None
        if output_path is not None:
            writer = ChunkedCacheWriter(
                output_path,
                n_layers=n_layers,
                per_head=per_head,
                split=split,
                batch_offset=batch_offset,
                dtype=cache_dtype,
            )

        pbar = tqdm(total=max_batches, desc="Phase B | cache (all layers)")
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids, attn_mask = _to_device(batch, device)

            out = model(
                input_ids,
                attn_mask=attn_mask,
                return_hidden_states=True,
                return_pre_attention_hidden=True,
                return_attentions=True,
                return_per_head_attention=per_head,
            )
            layer_payload: dict[int, dict[str, torch.Tensor]] = {}
            for li in range(n_layers):
                attn = out["attentions"][li]
                if not per_head and attn.dim() == 4:
                    attn = attn.mean(dim=1)
                layer_payload[li] = {
                    "hidden_states": out["pre_attention_hidden"][li],
                    "attention": attn,
                }
            if writer is not None:
                writer.write_batch(i, layer_payload)
            pbar.update(1)
        pbar.close()

        manifest = writer.finalize() if writer is not None else {}
        result = {
            "format": "chunked_attention_v1",
            "split": split,
            "batch_offset": batch_offset,
            "num_batches": max_batches,
            "num_sequences": manifest.get("num_sequences", 0),
            "cache_dir": str(output_path) if output_path else None,
        }
        return result

    all_hidden = []
    all_attention = []

    layer_label = layer_idx if layer_idx >= 0 else model.n_layers + layer_idx
    pbar = tqdm(total=max_batches, desc=f"Collect attention cache (layer {layer_label})")
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids, attn_mask = _to_device(batch, device)

        data = model.collect_attention_data(
            input_ids, attn_mask, layer_idx=layer_idx, per_head=per_head
        )
        all_hidden.append(data["hidden_states"].detach().cpu())
        all_attention.append(data["attention"].detach().cpu())
        pbar.update(1)
    pbar.close()

    hidden = torch.cat(all_hidden, dim=0)
    attention = torch.cat(all_attention, dim=0)
    layer = layer_idx if layer_idx >= 0 else model.n_layers + layer_idx
    result = {
        "hidden_states": hidden,
        "attention": attention,
        "layer_idx": layer,
        "per_head": per_head,
        "split": split,
        "batch_offset": batch_offset,
        "num_batches": max_batches,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        meta = {
            "num_batches": len(all_hidden),
            "hidden_shape": list(hidden.shape),
            "layer_idx": layer,
            "per_head": per_head,
            "split": split,
            "batch_offset": batch_offset,
            "num_sequences": int(hidden.shape[0]),
        }
        with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return result


def train_per_layer_routers(
    router: PerLayerRouter,
    cache_path: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    runner,
    metrics_logger,
    logger,
    config: dict,
    eval_fn: Callable | None = None,
    holdout_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Train each layer's router on that layer's cached hidden/attention data."""
    from routing_attention.data.chunked_cache import is_chunked_cache

    router_cfg = config["router"]
    results = {}

    if is_chunked_cache(cache_path):
        trainer_kwargs = dict(
            device=device,
            runner=runner,
            metrics_logger=metrics_logger,
            logger=logger,
            loss_type=router_cfg["loss_type"],
            top_k=router_cfg["top_k"],
            temperature=router_cfg["temperature"],
            max_steps=router_cfg["max_steps"],
            eval_every=resolve_eval_every(config, "router"),
            save_every=router_cfg["save_every"],
            batch_size=router_cfg.get("batch_size", 4),
            num_negatives=router_cfg.get("num_negatives", 64),
            eval_fn=eval_fn,
            attention_supervision=config.get("data_collection", {}).get("attention_supervision", "head_avg"),
            eval_max_samples=config.get("validation", {}).get("eval_max_samples", 8),
            use_amp=config.get("training", {}).get("use_amp", True),
        )
        skip_completed = bool(
            config.get("index_pretrain", {}).get("skip_completed_layers", True)
        )
        layer_bar = tqdm(range(router.n_layers), desc="Per-layer router training", unit="layer")
        for layer_idx in layer_bar:
            layer_bar.set_postfix(layer=layer_idx)
            layer_router = router.get_router(layer_idx)
            orig_dir = runner.checkpoint_dir
            layer_ckpt_dir = orig_dir / f"layer_{layer_idx}"
            final_ckpt = layer_ckpt_dir / "final.pt"
            if skip_completed and final_ckpt.exists():
                from routing_attention.utils.checkpoint import load_checkpoint

                load_checkpoint(final_ckpt, layer_router, device=device, strict=False)
                logger.info(
                    "Skipping layer %d / %d — final checkpoint exists (%s)",
                    layer_idx,
                    router.n_layers - 1,
                    final_ckpt,
                )
                results[layer_idx] = {
                    "skipped": True,
                    "checkpoint": str(final_ckpt),
                    "steps": router_cfg["max_steps"],
                    "layer_idx": layer_idx,
                }
                continue
            logger.info("Training router for layer %d / %d", layer_idx, router.n_layers - 1)
            layer_opt = torch.optim.AdamW(layer_router.parameters(), lr=router_cfg["lr"])
            runner.checkpoint_dir = layer_ckpt_dir
            runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer = RouterTrainer(router=layer_router, optimizer=layer_opt, layer_idx=layer_idx, **trainer_kwargs)
            results[layer_idx] = trainer.train_on_chunked_cache(
                cache_path,
                holdout_cache_path=holdout_cache_path,
                layer_idx=layer_idx,
            )
            runner.checkpoint_dir = orig_dir
        save_checkpoint(
            runner.checkpoint_dir / "best.pt",
            router,
            optimizer,
            extra={"per_layer": True, "layer_results": results},
        )
        return results

    data = torch.load(cache_path, map_location=device, weights_only=False)
    holdout_data = (
        torch.load(holdout_cache_path, map_location=device, weights_only=False)
        if holdout_cache_path is not None and holdout_cache_path.exists()
        else None
    )

    if "layers" not in data:
        raise ValueError("Per-layer training requires all_layers cache. Set data_collection.all_layers: true")

    trainer_kwargs = dict(
        device=device,
        runner=runner,
        metrics_logger=metrics_logger,
        logger=logger,
        loss_type=router_cfg["loss_type"],
        top_k=router_cfg["top_k"],
        temperature=router_cfg["temperature"],
        max_steps=router_cfg["max_steps"],
        eval_every=resolve_eval_every(config, "router"),
        save_every=router_cfg["save_every"],
        batch_size=router_cfg.get("batch_size", 4),
        num_negatives=router_cfg.get("num_negatives", 64),
        eval_fn=eval_fn,
        attention_supervision=config.get("data_collection", {}).get("attention_supervision", "head_avg"),
        eval_max_samples=config.get("validation", {}).get("eval_max_samples", 8),
        use_amp=config.get("training", {}).get("use_amp", True),
    )

    skip_completed = bool(config.get("index_pretrain", {}).get("skip_completed_layers", True))
    layer_bar = tqdm(range(router.n_layers), desc="Per-layer router training", unit="layer")
    for layer_idx in layer_bar:
        layer_bar.set_postfix(layer=layer_idx)
        layer_router = router.get_router(layer_idx)
        orig_dir = runner.checkpoint_dir
        layer_ckpt_dir = orig_dir / f"layer_{layer_idx}"
        final_ckpt = layer_ckpt_dir / "final.pt"
        if skip_completed and final_ckpt.exists():
            from routing_attention.utils.checkpoint import load_checkpoint

            load_checkpoint(final_ckpt, layer_router, device=device, strict=False)
            logger.info(
                "Skipping layer %d / %d — final checkpoint exists (%s)",
                layer_idx,
                router.n_layers - 1,
                final_ckpt,
            )
            results[layer_idx] = {
                "skipped": True,
                "checkpoint": str(final_ckpt),
                "steps": router_cfg["max_steps"],
                "layer_idx": layer_idx,
            }
            continue
        logger.info("Training router for layer %d / %d", layer_idx, router.n_layers - 1)
        layer_cache = data["layers"][layer_idx]
        hidden = layer_cache["hidden_states"].to(device, non_blocking=True).contiguous()
        attention = layer_cache["attention"].to(device, non_blocking=True).contiguous()
        eval_hidden, eval_attention = None, None
        if holdout_data is not None and "layers" in holdout_data:
            holdout_layer = holdout_data["layers"][layer_idx]
            eval_hidden = holdout_layer["hidden_states"].to(device, non_blocking=True).contiguous()
            eval_attention = holdout_layer["attention"].to(device, non_blocking=True).contiguous()

        layer_router = router.get_router(layer_idx)
        layer_opt = torch.optim.AdamW(layer_router.parameters(), lr=router_cfg["lr"])

        orig_dir = runner.checkpoint_dir
        runner.checkpoint_dir = orig_dir / f"layer_{layer_idx}"
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer = RouterTrainer(router=layer_router, optimizer=layer_opt, layer_idx=layer_idx, **trainer_kwargs)
        results[layer_idx] = trainer.train_on_tensors(
            hidden,
            attention,
            layer_idx=layer_idx,
            eval_hidden=eval_hidden,
            eval_attention=eval_attention,
        )
        runner.checkpoint_dir = orig_dir

    save_checkpoint(
        runner.checkpoint_dir / "best.pt",
        router,
        optimizer,
        extra={"per_layer": True, "layer_results": results},
    )
    return results


def train_per_layer_addresses(
    address_book: PerLayerAddressBook,
    cache_path: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    runner,
    metrics_logger,
    logger,
    config: dict,
    eval_fn: Callable | None = None,
    holdout_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Train per-layer learned address projections (Phase C for learned_address mode)."""
    return train_per_layer_routers(
        router=address_book,
        cache_path=cache_path,
        optimizer=optimizer,
        device=device,
        runner=runner,
        metrics_logger=metrics_logger,
        logger=logger,
        config=config,
        eval_fn=eval_fn,
        holdout_cache_path=holdout_cache_path,
    )
