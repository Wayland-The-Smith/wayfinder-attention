"""
Fused sparse multi-head attention (Layer 3).

Pipeline: causal top-K on search vectors -> gather K/V -> softmax -> output.
Search space (R) and meat space (H, D) may differ — same kernel family for
routing, learned-address, and key-vector (head-mean search) variants.

Performance notes:
  The legacy per-token Triton kernel launched B*H*T programs (~28s @ T=2048).
  Production default is ``batched``: vectorized gather + batched matmul (~2.5ms
  meat @ T=16k). ``triton_head`` (B*H programs, tl.range over T) is kept for
  correctness checks only — irregular K/V gather is faster via PyTorch CUDA gather.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from routing_attention.kernels.causal_topk import causal_topk
from routing_attention.kernels._common import ensure_cuda_contiguous, has_triton, next_power_of_2

_MAX_TRITON_K = 64
_MIN_TRITON_D = 16
_MAX_TRITON_D = 128


@dataclass
class SparseMeatConfig:
    """Launch tuning for Layer-3 sparse meat attention."""

    method: str = "auto"  # auto | batched | triton_head | reference
    block_d: int | None = None  # constexpr D tile for triton_head; default next_pow2(D)
    k_pad: int | None = None  # constexpr K tile for triton_head; default next_pow2(K)
    num_warps: int = 4
    compute_dtype: str | None = None  # None | float16 | bfloat16 for batched matmul


def _resolve_sparse_meat_config(
    T: int,
    H: int,
    D: int,
    k_eff: int,
    cfg: SparseMeatConfig | None,
) -> SparseMeatConfig:
    cfg = cfg or SparseMeatConfig()
    k_pad = cfg.k_pad or min(next_power_of_2(max(k_eff, 1)), _MAX_TRITON_K)
    block_d = cfg.block_d or min(next_power_of_2(max(D, _MIN_TRITON_D)), _MAX_TRITON_D)
    return SparseMeatConfig(
        method=cfg.method,
        block_d=block_d,
        k_pad=k_pad,
        num_warps=cfg.num_warps,
        compute_dtype=cfg.compute_dtype,
    )


def fused_sparse_attention_available() -> bool:
    return has_triton()


@torch.no_grad()
def _gather_kv_batched(
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V at shared indices (B,T,K) for all heads -> (B,H,T,K,D)."""
    B, H, T, D = k.shape
    b_idx = torch.arange(B, device=k.device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=k.device).view(1, H, 1, 1)
    idx = indices.long().unsqueeze(1)
    k_sel = k[b_idx, h_idx, idx, :]
    v_sel = v[b_idx, h_idx, idx, :]
    return k_sel, v_sel


@torch.no_grad()
def sparse_meat_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Baseline sparse meat: gather + softmax + weighted sum."""
    k_sel, v_sel = _gather_kv_batched(k, v, indices)
    scores = torch.matmul(
        q.unsqueeze(-2), k_sel.transpose(-2, -1)
    ).squeeze(-2) * scale
    attn = F.softmax(scores, dim=-1)
    return (attn.unsqueeze(-1) * v_sel).sum(dim=-2)


@torch.no_grad()
def _sparse_meat_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    *,
    compute_dtype: str | None = None,
) -> torch.Tensor:
    """
    Vectorized meat attention — few cuBLAS-style kernels, no per-token launches.

    Uses batched matmul for scores and output projection. Default production path.
    """
    out_dtype = q.dtype
    if compute_dtype == "float16":
        cast = torch.float16
    elif compute_dtype == "bfloat16":
        cast = torch.bfloat16
    else:
        cast = None

    if cast is not None:
        q, k, v = q.to(cast), k.to(cast), v.to(cast)

    k_sel, v_sel = _gather_kv_batched(k, v, indices)
    # (B,H,T,1,D) @ (B,H,T,D,K) -> (B,H,T,1,K)
    scores = torch.matmul(q.unsqueeze(-2), k_sel.transpose(-2, -1)).squeeze(-2) * scale
    attn = F.softmax(scores, dim=-1)
    # (B,H,T,1,K) @ (B,H,T,K,D) -> (B,H,T,1,D)
    return torch.matmul(attn.unsqueeze(-2), v_sel).squeeze(-2).to(out_dtype)


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _sparse_meat_head_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        IDX_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_ib,
        stride_it,
        stride_ik,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        scale,
        T,
        D,
        K_TOP: tl.constexpr,
        K_EFF: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per (batch, head). Stream query positions with tl.range(T).

        Amortizes launch overhead: B*H programs instead of B*H*T.
        """
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        k_offs = tl.arange(0, K_TOP)

        q_base = Q_ptr + batch_id * stride_qb + head_id * stride_qh
        k_base = K_ptr + batch_id * stride_kb + head_id * stride_kh
        v_base = V_ptr + batch_id * stride_vb + head_id * stride_vh
        idx_base = IDX_ptr + batch_id * stride_ib
        out_base = Out_ptr + batch_id * stride_ob + head_id * stride_oh

        for t in tl.range(0, T):
            q_ptrs = q_base + t * stride_qt + d_offs * stride_qd
            q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

            scores = tl.full([K_TOP], float("-inf"), tl.float32)
            for ki in tl.static_range(K_TOP):
                active = ki < K_EFF
                j = tl.load(idx_base + t * stride_it + ki * stride_ik).to(tl.int32)
                j = tl.maximum(j, 0)
                j = tl.minimum(j, T - 1)
                k_ptrs = k_base + j * stride_kt + d_offs * stride_kd
                k_vec = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                dot = tl.sum(q_vec * k_vec, axis=0) * scale
                dot = tl.where(active, dot, float("-inf"))
                scores = tl.where(k_offs == ki, dot, scores)

            max_s = tl.max(scores, axis=0)
            exp_s = tl.exp(scores - max_s)
            denom = tl.sum(exp_s, axis=0)
            weights = exp_s / denom

            out_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
            for ki in tl.static_range(K_TOP):
                active = ki < K_EFF
                j = tl.load(idx_base + t * stride_it + ki * stride_ik).to(tl.int32)
                j = tl.maximum(j, 0)
                j = tl.minimum(j, T - 1)
                w = tl.where(active, weights[ki], 0.0)
                v_ptrs = v_base + j * stride_vt + d_offs * stride_vd
                v_vec = tl.load(v_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                out_vec = out_vec + w * v_vec

            out_ptrs = out_base + t * stride_ot + d_offs * stride_od
            tl.store(out_ptrs, out_vec, mask=d_mask)

    def _sparse_meat_triton_head(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
        scale: float,
        cfg: SparseMeatConfig,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        k_eff = indices.shape[-1]
        q, k, v, indices = ensure_cuda_contiguous(q, k, v, indices)

        resolved = _resolve_sparse_meat_config(T, H, D, k_eff, cfg)
        k_pad = resolved.k_pad
        block_d = resolved.block_d

        out = torch.empty_like(q, dtype=q.dtype)
        grid = (B, H)
        _sparse_meat_head_kernel[grid](
            q,
            k,
            v,
            indices,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            scale,
            T,
            D,
            K_TOP=k_pad,
            K_EFF=k_eff,
            BLOCK_D=block_d,
            num_warps=resolved.num_warps,
        )
        return out


@torch.no_grad()
def sparse_meat_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    *,
    method: str = "auto",
    config: SparseMeatConfig | None = None,
) -> torch.Tensor:
    """
    Multi-head sparse attention on pre-selected neighbor indices.

    Args:
        q: (B, H, T, D)
        k: (B, H, T, D)
        v: (B, H, T, D)
        indices: (B, T, K) int32 — shared across heads
        method: auto | triton_head | batched | reference
        config: optional launch tuning (block_d, k_pad, num_warps, triton_min_seq)
    """
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    cfg = config or SparseMeatConfig()
    B, H, T, D = q.shape
    k_eff = indices.shape[-1]
    resolved = _resolve_sparse_meat_config(T, H, D, k_eff, cfg)

    effective = method if method != "auto" else cfg.method
    if effective == "auto":
        # Batched PyTorch is the reliable default; triton_head available via config/method.
        if q.is_cuda:
            effective = "batched"
        else:
            effective = "reference"

    if effective == "triton_head" and q.is_cuda and has_triton():
        try:
            return _sparse_meat_triton_head(q, k, v, indices, scale, resolved)
        except Exception:
            effective = "batched"

    if effective == "batched" and q.is_cuda:
        return _sparse_meat_batched(
            q, k, v, indices, scale, compute_dtype=resolved.compute_dtype
        )

    return sparse_meat_attention_reference(q, k, v, indices, scale)


@torch.no_grad()
def fused_sparse_attention(
    q_search: torch.Tensor,
    k_search: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
    *,
    retrieval_method: str = "auto",
    meat_method: str = "auto",
    meat_config: SparseMeatConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full Layer-3 path: causal top-K retrieval + sparse meat attention.

    Args:
        q_search, k_search: (B, T, R) search-space vectors
        q, k, v: (B, H, T, D) meat projections
        meat_method: auto | triton_head | batched | reference
        meat_config: optional SparseMeatConfig for block_d / k_pad / num_warps
    Returns:
        out (B, H, T, D), indices (B, T, K) int32
    """
    indices = causal_topk(q_search, k_search, top_k, method=retrieval_method)
    out = sparse_meat_attention(
        q, k, v, indices, scale, method=meat_method, config=meat_config
    )
    return out, indices
