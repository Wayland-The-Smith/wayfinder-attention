"""Language modeling evaluation metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate model on LM loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    digit_correct = 0.0
    digit_total = 0
    track_digit = hasattr(model, "digit_head") and model.digit_head is not None

    non_blocking = device.type == "cuda"
    pbar = tqdm(total=max_batches, desc="Evaluate LM", leave=False) if max_batches else None
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        attn_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device, non_blocking=non_blocking)
        if labels is not None:
            labels = labels.to(device, non_blocking=non_blocking)

        out = model(input_ids, attn_mask=attn_mask, labels=labels if track_digit else None)
        loss = out["loss"]
        n_tokens = (attn_mask[:, 1:].sum() if attn_mask is not None else input_ids.numel())

        total_loss += loss.item() * n_tokens.item()
        total_tokens += n_tokens.item()
        if track_digit and labels is not None and "digit_accuracy" in out:
            n = labels.shape[0]
            digit_correct += out["digit_accuracy"].item() * n
            digit_total += n
        batches += 1
        if pbar is not None:
            pbar.update(1)
        if max_batches and batches >= max_batches:
            break
    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / max(total_tokens, 1)
    metrics: dict[str, float] = {
        "loss": avg_loss,
        "perplexity": compute_perplexity(avg_loss),
        "total_tokens": total_tokens,
        "batches_evaluated": batches,
    }
    if digit_total > 0:
        metrics["digit_accuracy"] = digit_correct / digit_total

    return metrics


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 100))


def compare_attention_types(
    results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build comparison table from multiple attention type evaluations."""
    baseline_key = "dense"
    if baseline_key not in results:
        baseline_key = next(iter(results))

    baseline = results[baseline_key]
    comparison = {"baseline": baseline_key, "models": results, "relative": {}}

    for name, metrics in results.items():
        if name == baseline_key:
            continue
        comparison["relative"][name] = {
            "loss_delta": metrics["loss"] - baseline["loss"],
            "ppl_ratio": metrics["perplexity"] / max(baseline["perplexity"], 1e-9),
        }
    return comparison
