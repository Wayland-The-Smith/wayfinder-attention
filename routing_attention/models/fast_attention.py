"""
Production attention kernels for Experiment 7 head-to-head comparisons.

Linear attention (flash-linear-attention / FLA):
  https://github.com/fla-org/flash-linear-attention

  Kernels (FLA 0.4+):
    - ``chunk_linear_attn`` — chunkwise parallel (default; fastest at H=4, D=64, T=8k on RTX 5090)
    - ``fused_chunk_linear_attn`` — lower HBM traffic; better on very large heads/dim
    - ``fused_recurrent_linear_attn`` — recurrent fused path (slower for training-length T)

Sliding window: PyTorch Flex Attention (compiled).
Dense: PyTorch SDPA / Flash (``DenseSDPAAttention``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

FLA_LINEAR_KERNEL_ENV = "FLA_LINEAR_KERNEL"
FLA_LINEAR_KERNELS = ("chunk", "fused_chunk", "fused_recurrent", "auto")

_FLA_AVAILABLE: bool | None = None
_FLEX_AVAILABLE: bool | None = None
_RESOLVED_LINEAR_KERNEL: str | None = None


def has_fla() -> bool:
    global _FLA_AVAILABLE
    if _FLA_AVAILABLE is None:
        try:
            import fla.ops.linear_attn  # noqa: F401

            _FLA_AVAILABLE = True
        except ImportError:
            _FLA_AVAILABLE = False
    return _FLA_AVAILABLE


def has_flex_attention() -> bool:
    global _FLEX_AVAILABLE
    if _FLEX_AVAILABLE is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

            _FLEX_AVAILABLE = True
        except ImportError:
            _FLEX_AVAILABLE = False
    return _FLEX_AVAILABLE


def require_fla() -> None:
    if not has_fla():
        raise ImportError(
            "flash-linear-attention (package `fla`) is required for the linear attention baseline. "
            "Install: pip install flash-linear-attention"
        )


def require_flex_attention() -> None:
    if not has_flex_attention():
        raise ImportError(
            "PyTorch Flex Attention (torch>=2.4) is required for sliding-window attention. "
            "Upgrade PyTorch or use a CUDA build with flex_attention support."
        )


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def _to_bthd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    """ELU+1 feature map; return (B, T, H, D) tensors for FLA."""
    q_feat = _elu_plus_one(q * scale)
    k_feat = _elu_plus_one(k)
    return (
        q_feat.transpose(1, 2).contiguous(),
        k_feat.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )


def _run_chunk(q_bthd, k_bthd, v_bthd) -> torch.Tensor:
    from fla.ops.linear_attn import chunk_linear_attn

    out, _ = chunk_linear_attn(
        q_bthd, k_bthd, v_bthd, scale=1.0, head_first=False, normalize=True
    )
    return out


def _run_fused_chunk(q_bthd, k_bthd, v_bthd) -> torch.Tensor:
    from fla.ops.linear_attn import fused_chunk_linear_attn

    out, _ = fused_chunk_linear_attn(q_bthd, k_bthd, v_bthd, scale=1.0, normalize=True)
    return out


def _run_fused_recurrent(q_bthd, k_bthd, v_bthd) -> torch.Tensor:
    from fla.ops.linear_attn import fused_recurrent_linear_attn

    out, _ = fused_recurrent_linear_attn(q_bthd, k_bthd, v_bthd, scale=1.0, normalize=True)
    return out


_KERNEL_RUNNERS: dict[str, Callable] = {
    "chunk": _run_chunk,
    "fused_chunk": _run_fused_chunk,
    "fused_recurrent": _run_fused_recurrent,
}


@torch.no_grad()
def _microbench_kernel(name: str, device: torch.device, dtype: torch.dtype) -> float:
    """One fwd+bwd timing for kernel selection (auto mode)."""
    B, H, T, D = 1, 4, 2048, 64
    runner = _KERNEL_RUNNERS[name]
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    scale = D**-0.5
    for _ in range(2):
        q_b, k_b, v_b = _to_bthd(q, k, v, scale)
        out = runner(q_b, k_b, v_b).transpose(1, 2)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    q_b, k_b, v_b = _to_bthd(q, k, v, scale)
    out = runner(q_b, k_b, v_b).transpose(1, 2)
    out.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def resolve_fla_linear_kernel(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """
    Pick FLA linear kernel.

    ``auto`` micro-benchmarks ``chunk`` vs ``fused_chunk`` once per process (training dtype).
    For our 6×256 trunk (H=4, D=64), ``chunk`` consistently wins on RTX 5090.
    """
    global _RESOLVED_LINEAR_KERNEL
    if _RESOLVED_LINEAR_KERNEL is not None:
        return _RESOLVED_LINEAR_KERNEL

    requested = os.environ.get(FLA_LINEAR_KERNEL_ENV, "auto").strip().lower()
    if requested in _KERNEL_RUNNERS:
        _RESOLVED_LINEAR_KERNEL = requested
        return _RESOLVED_LINEAR_KERNEL

    if requested != "auto":
        raise ValueError(
            f"Unknown {FLA_LINEAR_KERNEL_ENV}={requested!r}; "
            f"use one of {FLA_LINEAR_KERNELS}"
        )

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        _RESOLVED_LINEAR_KERNEL = "chunk"
        return _RESOLVED_LINEAR_KERNEL

    try:
        t_chunk = _microbench_kernel("chunk", dev, dtype)
        t_fused = _microbench_kernel("fused_chunk", dev, dtype)
        _RESOLVED_LINEAR_KERNEL = "chunk" if t_chunk <= t_fused else "fused_chunk"
    except Exception:
        _RESOLVED_LINEAR_KERNEL = "chunk"
    return _RESOLVED_LINEAR_KERNEL


def fla_causal_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    kernel: str | None = None,
) -> torch.Tensor:
    """
    Causal linear attention (ELU+1) via FLA Triton kernels.

    Args:
        q, k, v: (B, n_heads, T, head_dim)
    """
    require_fla()
    kernel_name = kernel or resolve_fla_linear_kernel(device=q.device, dtype=q.dtype)
    runner = _KERNEL_RUNNERS[kernel_name]
    q_bthd, k_bthd, v_bthd = _to_bthd(q, k, v, scale)
    out_bthd = runner(q_bthd, k_bthd, v_bthd)
    return out_bthd.transpose(1, 2)


def warmup_fla_linear_kernels(
    *,
    device: torch.device,
    n_heads: int = 4,
    head_dim: int = 64,
    context_length: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """
    Pre-compile FLA Triton kernels before the timed training loop.

    Avoids a multi-second (or minute-long) first-step compile spike in tqdm.
  """
    kernel = resolve_fla_linear_kernel(device=device, dtype=dtype)
    B = 1
    q = torch.randn(B, n_heads, context_length, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, n_heads, context_length, head_dim, device=device, dtype=dtype)
    v = torch.randn(B, n_heads, context_length, head_dim, device=device, dtype=dtype)
    scale = head_dim**-0.5
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    out = fla_causal_linear_attention(q, k, v, scale, kernel=kernel)
    out.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return kernel


@lru_cache(maxsize=8)
def _compiled_flex_attention() -> Callable:
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention)


@lru_cache(maxsize=64)
def _sliding_window_block_mask(T: int, window_size: int, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=T,
        KV_LEN=T,
        device=device,
    )


def flex_sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scale: float,
    dropout: nn.Dropout,
) -> torch.Tensor:
    """Causal sliding-window softmax attention via compiled Flex Attention."""
    require_flex_attention()
    q = q * scale
    block_mask = _sliding_window_block_mask(q.shape[2], window_size, q.device)
    flex_fn = _compiled_flex_attention()
    out = flex_fn(q, k, v, block_mask=block_mask)
    drop_p = dropout.p if dropout.training else 0.0
    if drop_p > 0:
        out = F.dropout(out, drop_p, training=True)
    return out


def warmup_flex_sliding_window(
    *,
    device: torch.device,
    n_heads: int = 4,
    head_dim: int = 64,
    context_length: int = 2048,
    window_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Pre-compile Flex Attention block mask + kernel."""
    q = torch.randn(1, n_heads, context_length, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, n_heads, context_length, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, n_heads, context_length, head_dim, device=device, dtype=dtype)
    drop = nn.Dropout(0.0)
    flex_sliding_window_attention(q, k, v, window_size, head_dim**-0.5, drop)
    if device.type == "cuda":
        torch.cuda.synchronize()


def backend_status() -> dict[str, bool | str]:
    kernel: str = _RESOLVED_LINEAR_KERNEL or ("auto" if has_fla() else "unavailable")
    return {
        "fla_linear": has_fla(),
        "fla_linear_kernel": kernel,
        "flex_sliding_window": has_flex_attention(),
        "sdpa_flash": bool(
            torch.cuda.is_available()
            and getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)()
        ),
    }
