"""Shared helpers for Triton kernel launch configuration."""

from __future__ import annotations

import torch


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def promote_compute_dtype(
    q: torch.Tensor,
    k: torch.Tensor,
    dtype: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Pick a stable compute dtype for dot products."""
    if dtype == "float16":
        target = torch.float16
    elif dtype == "bfloat16":
        target = torch.bfloat16
    elif dtype == "float32":
        target = torch.float32
    else:
        if q.dtype in (torch.float16, torch.bfloat16):
            target = q.dtype
        else:
            target = torch.float32
    return q.to(target), k.to(target), target


def ensure_cuda_contiguous(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.contiguous() if t.is_cuda else t for t in tensors)


def has_triton() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False
