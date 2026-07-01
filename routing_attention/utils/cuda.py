"""CUDA / GPU training performance configuration."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def configure_cuda_training(config: dict[str, Any] | None = None) -> dict[str, bool]:
    """
    Apply GPU settings tuned for modern NVIDIA cards (e.g. RTX 5090).

    Returns dict of which flags were enabled.
    """
    cfg = (config or {}).get("training", {})
    applied: dict[str, bool] = {}

    if not torch.cuda.is_available():
        return applied

    if cfg.get("cudnn_deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True
        applied["cublas_workspace_config"] = True
    elif cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
        applied["cudnn_benchmark"] = True

    if cfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["tf32"] = True

    if cfg.get("matmul_precision", "high") == "high":
        try:
            torch.set_float32_matmul_precision("high")
            applied["matmul_precision_high"] = True
        except Exception:
            pass

    return applied


def maybe_compile_model(model: torch.nn.Module, config: dict[str, Any] | None = None) -> torch.nn.Module:
    """Optionally wrap model with torch.compile (off by default — enable after dry-run)."""
    cfg = (config or {}).get("training", {})
    if not cfg.get("torch_compile", False):
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable — skipping")
        return model
    mode = cfg.get("compile_mode", "reduce-overhead")
    logger.info("Applying torch.compile(mode=%s)", mode)
    return torch.compile(model, mode=mode)
