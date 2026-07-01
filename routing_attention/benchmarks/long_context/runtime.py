"""Runtime / device verification for Experiment 7."""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn


def collect_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device_type": device.type,
        "device_index": device.index,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(idx))
        props = torch.cuda.get_device_properties(idx)
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def assert_expected_device(config: dict, device: torch.device) -> None:
    """Fail fast if config requests CUDA but we fell back to CPU."""
    requested = config.get("device", "cuda")
    if requested == "cuda" and device.type != "cuda":
        raise RuntimeError(
            "Config requests device=cuda but runtime is on CPU. "
            "Run in WSL with conda env fla311 and an NVIDIA GPU."
        )


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    if a.type == "cpu":
        return True
    return (a.index or 0) == (b.index or 0)


def verify_model_on_device(model: nn.Module, device: torch.device) -> dict[str, Any]:
    params = list(model.parameters())
    if not params:
        return {"param_device": str(device), "n_params": 0}
    param_device = params[0].device
    if not _same_device(param_device, device):
        raise RuntimeError(
            f"Model parameters on {param_device} but expected {device}"
        )
    n_params = sum(p.numel() for p in params)
    return {"param_device": str(param_device), "n_params": n_params}


def peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_peak_vram(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
