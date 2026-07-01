"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint with optional optimizer state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load checkpoint into model (and optionally optimizer)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
