"""
Fused causal top-K inner-product retrieval (Layer 2).

Computes top-K key indices j <= i maximizing dot(query[i], key[j]) WITHOUT
materializing a full (B, T, T) score matrix.

Why brute_force can be beaten at long T:
  brute_force writes ~T² floats to HBM (mask + topk readback). Fused paths
  stream tiles and keep only (B, T, K) state — same FLOPs, far less bandwidth.

Paths (method=fused_causal always uses these — never silent cuBLAS redirect):
  1. row_blocked  — triangular Q-row tiles + topk, no T×T buffer (default, fastest)
  2. heap_gemm    — K-tile stream inside row-block, narrow heap merge (streaming)
  3. triton_heap  — Triton per-row heap (opt-in via prefer_triton / method=triton)
  4. key_tiled    — full-row K-tile streaming (OOM fallback)
"""

from __future__ import annotations

import torch

from routing_attention.kernels._common import (
    ensure_cuda_contiguous,
    has_triton,
    next_power_of_2,
    promote_compute_dtype,
)

_MAX_TRITON_K = 64
_MIN_TRITON_R = 16


def causal_topk_available() -> bool:
    return has_triton()


@torch.no_grad()
def _finalize_causal_topk(
    out_val: torch.Tensor,
    out_idx: torch.Tensor,
    k_eff: int,
) -> torch.Tensor:
    """
    Fix padding indices for -inf slots (rows with fewer than k_eff causal keys).

    Fully vectorized — no .item()/.tolist() GPU sync (was ~100ms/call).
    Invalid slots get non-colliding pad indices; scores are -inf so softmax ignores them.
    """
    idx = out_idx[..., :k_eff].long()
    vals = out_val[..., :k_eff]
    valid = vals.isfinite() & (vals > -1e30)

    T = out_val.shape[1]
    rows = torch.arange(T, device=out_val.device).view(1, T, 1)
    slots = torch.arange(k_eff, device=out_val.device).view(1, 1, k_eff)
    pad_idx = (rows + k_eff + slots + 1) % T
    return torch.where(valid, idx, pad_idx).to(torch.int32)


def causal_topk_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Baseline: full GEMM + causal mask + topk (materializes T×T)."""
    B, T, R = query.shape
    scores = torch.bmm(query.float(), key.float().transpose(1, 2))
    causal = torch.tril(torch.ones(T, T, device=query.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    k_eff = min(top_k, T)
    _, indices = torch.topk(scores, k=k_eff, dim=-1)
    return indices.to(torch.int32)


@torch.no_grad()
def _pick_row_block(T: int) -> int:
    if T >= 16384:
        return 4096
    if T >= 8192:
        return 2048
    if T >= 2048:
        return 1024
    return 512


@torch.no_grad()
def _pick_micro_blocks(T: int) -> tuple[int, int]:
    """Row-block × K-tile sizes for heap-during-GEMM (narrow merge width K+BJ)."""
    if T >= 16384:
        return 4096, 4096
    if T >= 8192:
        return 2048, 2048
    if T >= 2048:
        return 1024, 1024
    return 512, 512


@torch.no_grad()
def _pick_key_block(T: int) -> int:
    """Wide K tiles → few merges; each merge scans only K+block_j columns per row."""
    if T >= 16384:
        return 4096
    if T >= 8192:
        return 2048
    if T >= 2048:
        return 1024
    return 512


@torch.no_grad()
def _apply_causal_mask_scores(
    scores: torch.Tensor,
    i0: int,
    i1: int,
    *,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask scores (B, bm, i1) so only keys j <= row remain finite."""
    bm = i1 - i0
    if i0 == 0 and bm == i1:
        tri = causal_mask
        if tri is None or tri.shape != (bm, i1):
            tri = torch.tril(
                torch.ones(bm, i1, device=scores.device, dtype=torch.bool)
            )
        return scores.masked_fill(~tri.unsqueeze(0), float("-inf"))
    rows = torch.arange(i0, i1, device=scores.device).view(bm, 1)
    cols = torch.arange(0, i1, device=scores.device).view(1, i1)
    return scores.masked_fill((cols > rows).unsqueeze(0), float("-inf"))


@torch.no_grad()
def _causal_topk_row_blocked(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    block_m: int | None = None,
) -> torch.Tensor:
    """
    Fast fused path: triangular GEMM per Q row-block, topk per tile.

    Keys truncated to i1 for rows [i0, i1) (causal). Peak memory
    O(B × block_m × i1) ≤ O(B × block_m × T), never O(B × T × T).
    """
    B, T, R = query.shape
    k_eff = min(top_k, T)
    device = query.device
    block_m = block_m or _pick_row_block(T)

    q = query.float()
    k = key.float()

    out_val = torch.empty(B, T, k_eff, device=device, dtype=torch.float32)
    out_idx = torch.empty(B, T, k_eff, dtype=torch.int32, device=device)
    tri0: torch.Tensor | None = None

    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        bm = i1 - i0
        scores = torch.bmm(q[:, i0:i1], k[:, :i1].transpose(1, 2).contiguous())
        if i0 == 0 and bm == i1:
            if tri0 is None:
                tri0 = torch.tril(torch.ones(bm, i1, device=device, dtype=torch.bool))
            scores = _apply_causal_mask_scores(scores, i0, i1, causal_mask=tri0)
        else:
            scores = _apply_causal_mask_scores(scores, i0, i1)
        vals, idx = torch.topk(scores, k=k_eff, dim=-1, sorted=False)
        out_val[:, i0:i1] = vals
        out_idx[:, i0:i1] = idx.to(torch.int32)

    return _finalize_causal_topk(out_val, out_idx, k_eff)


@torch.no_grad()
def _causal_topk_heap_during_gemm(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    block_m: int | None = None,
    block_j: int | None = None,
) -> torch.Tensor:
    """
    Heap-during-GEMM: stream K tiles inside each Q row-block.

    Never materializes (BM, T) or (T, T). Each micro-step:
      GEMM (BM, R) × (R, BJ) → (BM, BJ) scores → merge into running top-K
    with topk width (K + BJ) ≈ 64 instead of T.

    Triangular: for rows [i0, i1) only keys [0, i1) are needed (causal bound).
    """
    B, T, R = query.shape
    k_eff = min(top_k, T)
    device = query.device
    bm_def, bj_def = _pick_micro_blocks(T)
    block_m = block_m or bm_def
    block_j = block_j or bj_def

    q = query.float()
    k = key.float()

    out_val = torch.empty(B, T, k_eff, device=device, dtype=torch.float32)
    out_idx = torch.empty(B, T, k_eff, dtype=torch.int32, device=device)
    merge_w = k_eff + block_j
    merge_val = torch.empty(B, block_m, merge_w, device=device, dtype=torch.float32)
    merge_idx = torch.empty(B, block_m, merge_w, device=device, dtype=torch.int32)

    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        bm = i1 - i0
        local_val = torch.full((B, bm, k_eff), float("-inf"), device=device)
        local_idx = torch.full((B, bm, k_eff), -1, dtype=torch.int32, device=device)
        row_ids = torch.arange(i0, i1, device=device).view(1, bm, 1)

        for j0 in range(0, i1, block_j):
            j1 = min(j0 + block_j, i1)
            bj = j1 - j0
            scores = torch.bmm(q[:, i0:i1], k[:, j0:j1].transpose(1, 2))
            cols = torch.arange(j0, j1, device=device).view(1, bj)
            scores = scores.masked_fill(cols > row_ids, float("-inf"))

            w = k_eff + bj
            merge_val[:, :bm, :k_eff] = local_val
            merge_val[:, :bm, k_eff:w] = scores
            merge_idx[:, :bm, :k_eff] = local_idx
            merge_idx[:, :bm, k_eff:w] = cols.to(torch.int32).expand(B, bm, bj)

            local_val, order = torch.topk(
                merge_val[:, :bm, :w], k=k_eff, dim=-1, sorted=False
            )
            local_idx = merge_idx[:, :bm, :w].gather(-1, order)

        out_val[:, i0:i1] = local_val
        out_idx[:, i0:i1] = local_idx.to(torch.int32)

    return _finalize_causal_topk(out_val, out_idx, k_eff)


# Alias for benchmarks / scripts
_causal_topk_heap_micro = _causal_topk_heap_during_gemm


@torch.no_grad()
def _merge_tile_topk(
    out_val: torch.Tensor,
    out_idx: torch.Tensor,
    scores: torch.Tensor,
    j0: int,
    k_eff: int,
    merge_buf_val: torch.Tensor,
    merge_buf_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge (B, T, block_j) scores into running top-K using preallocated buffers."""
    B, T, block_j = scores.shape
    device = scores.device
    j_idx = torch.arange(j0, j0 + block_j, device=device, dtype=torch.int32)
    row_ids = torch.arange(T, device=device).view(1, T, 1)
    scores = scores.masked_fill(row_ids < j_idx.view(1, 1, block_j), float("-inf"))

    w = k_eff + block_j
    merge_buf_val[:, :, :k_eff] = out_val
    merge_buf_val[:, :, k_eff:w] = scores
    merge_buf_idx[:, :, :k_eff] = out_idx
    merge_buf_idx[:, :, k_eff:w] = j_idx.view(1, 1, block_j).expand(B, T, block_j)

    new_val, order = torch.topk(merge_buf_val[:, :, :w], k=k_eff, dim=-1)
    new_idx = merge_buf_idx[:, :, :w].gather(-1, order)
    return new_val, new_idx


@torch.no_grad()
def _causal_topk_key_tiled(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    block_j: int | None = None,
) -> torch.Tensor:
    """K-tile streaming fallback — no T×T buffer."""
    B, T, R = query.shape
    k_eff = min(top_k, T)
    device = query.device
    block_j = block_j or min(4096, max(512, T // 8))

    q = query.to(torch.float16)
    k = key.to(torch.float16)

    out_idx = torch.full((B, T, k_eff), -1, dtype=torch.int32, device=device)
    out_val = torch.full((B, T, k_eff), float("-inf"), device=device, dtype=torch.float32)
    merge_val = torch.empty(B, T, k_eff + block_j, device=device, dtype=torch.float32)
    merge_idx = torch.empty(B, T, k_eff + block_j, device=device, dtype=torch.int32)

    for j0 in range(0, T, block_j):
        j1 = min(j0 + block_j, T)
        scores = torch.matmul(q, k[:, j0:j1, :].transpose(1, 2)).float()
        out_val, out_idx = _merge_tile_topk(
            out_val, out_idx, scores, j0, k_eff, merge_val, merge_idx
        )

    return _finalize_causal_topk(out_val, out_idx, k_eff)


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _topk_merge_scalar(top_val, top_idx, cand_val, cand_idx, K_TOP: tl.constexpr):
        offs = tl.arange(0, K_TOP)
        min_val = tl.min(top_val, axis=0)
        is_min = top_val == min_val
        min_pos = tl.min(tl.where(is_min, offs, K_TOP))
        replace = cand_val > min_val
        hit = offs == min_pos
        top_val = tl.where(hit, tl.where(replace, cand_val, top_val), top_val)
        top_idx = tl.where(hit, tl.where(replace, cand_idx, top_idx), top_idx)
        return top_val, top_idx

    @triton.jit
    def _causal_topk_row_kernel(
        Q_ptr,
        K_ptr,
        Out_val_ptr,
        Out_idx_ptr,
        stride_qb,
        stride_qt,
        stride_qr,
        stride_kb,
        stride_kt,
        stride_kr,
        stride_ovb,
        stride_ovt,
        stride_oib,
        stride_oit,
        T,
        R,
        K_TOP: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """One program per query row; stream K tiles, heap merge in registers."""
        batch_id = tl.program_id(0)
        row_i = tl.program_id(1)
        if row_i >= T:
            return

        r_offs = tl.arange(0, BLOCK_R)
        r_mask = r_offs < R
        j_offs = tl.arange(0, BLOCK_J)
        k_offs = tl.arange(0, K_TOP)

        q = tl.load(
            Q_ptr + batch_id * stride_qb + row_i * stride_qt + r_offs * stride_qr,
            mask=r_mask,
            other=0.0,
        ).to(tl.float32)

        top_val = tl.full([K_TOP], float("-inf"), tl.float32)
        top_idx = tl.full([K_TOP], -1, tl.int32)
        k_base = K_ptr + batch_id * stride_kb

        for j_base in tl.range(0, T, BLOCK_J):
            j_idx = j_base + j_offs
            j_mask = j_idx < T
            k_ptrs = k_base + j_idx[:, None] * stride_kt + r_offs[None, :] * stride_kr
            k_tile = tl.load(
                k_ptrs,
                mask=j_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(k_tile * q[None, :], axis=1)
            causal = j_idx <= row_i
            scores = tl.where(causal & j_mask, scores, float("-inf"))

            for jj in tl.static_range(BLOCK_J):
                cand_val = tl.sum(tl.where(j_offs == jj, scores, 0.0))
                top_val, top_idx = _topk_merge_scalar(
                    top_val, top_idx, cand_val, j_base + jj, K_TOP=K_TOP
                )

        tl.store(Out_val_ptr + batch_id * stride_ovb + row_i * stride_ovt + k_offs, top_val)
        tl.store(Out_idx_ptr + batch_id * stride_oib + row_i * stride_oit + k_offs, top_idx)

    def _pick_triton_blocks(T: int, R: int) -> tuple[int, int]:
        block_r = next_power_of_2(max(R, _MIN_TRITON_R))
        if T >= 8192:
            return 1024, block_r
        if T >= 2048:
            return 512, block_r
        return 256, block_r

    def _causal_topk_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        B, T, R = query.shape
        k_eff = min(top_k, T)
        if k_eff == 0:
            return torch.zeros(B, T, 0, dtype=torch.int32, device=query.device)

        query, key = ensure_cuda_contiguous(query, key)
        k_pad = min(next_power_of_2(max(k_eff, 1)), _MAX_TRITON_K)
        block_j, block_r = _pick_triton_blocks(T, R)

        out_val = torch.full((B, T, k_pad), float("-inf"), device=query.device, dtype=torch.float32)
        out_idx = torch.full((B, T, k_pad), -1, device=query.device, dtype=torch.int32)

        grid = (B, T)
        _causal_topk_row_kernel[grid](
            query,
            key,
            out_val,
            out_idx,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            out_val.stride(0),
            out_val.stride(1),
            out_idx.stride(0),
            out_idx.stride(1),
            T,
            R,
            K_TOP=k_pad,
            BLOCK_J=block_j,
            BLOCK_R=block_r,
            num_warps=4,
        )

        vals, order = torch.topk(out_val[..., :k_pad], k=k_eff, dim=-1)
        sorted_idx = out_idx[..., :k_pad].gather(-1, order)
        return _finalize_causal_topk(vals, sorted_idx, k_eff)


@torch.no_grad()
def _causal_topk_fused(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    *,
    prefer_triton: bool = False,
) -> torch.Tensor:
    """
    True fused causal top-K — never redirects to full T×T brute GEMM.

    Default: triangular row-blocked GEMM + topk (fastest on long T).
    Fallbacks: heap-during-GEMM (strict streaming), key-tiled (OOM).
    """
    k_eff = min(top_k, query.shape[1])

    if prefer_triton and has_triton() and query.is_cuda:
        try:
            return _causal_topk_triton(query, key, k_eff)
        except Exception:
            pass

    try:
        return _causal_topk_row_blocked(query, key, k_eff)
    except RuntimeError:
        try:
            return _causal_topk_heap_during_gemm(query, key, k_eff)
        except RuntimeError:
            return _causal_topk_key_tiled(
                query, key, k_eff, block_j=_pick_key_block(query.shape[1])
            )


@torch.no_grad()
def causal_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    top_k: int,
    *,
    method: str = "auto",
    dtype: str | None = None,
    prefer_triton: bool = False,
) -> torch.Tensor:
    """
    Causal top-K retrieval.

    Methods:
      fused_causal — triangular row-block fused GEMM (our vector-search kernel)
      triton       — Triton per-row heap-during-GEMM (long-T can be slow)
      tiled        — K-tile streaming fallback
      brute_force  — full T×T baseline for comparison only
    """
    if query.shape != key.shape:
        raise ValueError(f"query/key shape mismatch: {query.shape} vs {key.shape}")
    if query.dim() != 3:
        raise ValueError(f"expected (B,T,R), got {tuple(query.shape)}")

    B, T, R = query.shape
    k_eff = min(top_k, T)
    if k_eff == 0:
        return torch.zeros(B, T, 0, dtype=torch.int32, device=query.device)

    q, k, _ = promote_compute_dtype(query, key, dtype)

    if method == "auto":
        method = "fused_causal"

    if method == "fused_causal":
        return _causal_topk_fused(q, k, k_eff, prefer_triton=prefer_triton)

    if method == "triton":
        if not has_triton() or not q.is_cuda:
            raise RuntimeError("Triton CUDA kernel unavailable")
        return _causal_topk_triton(q, k, k_eff)

    if method in ("tiled", "streaming"):
        return _causal_topk_key_tiled(q, k, k_eff)

    if method == "brute_force":
        try:
            return causal_topk_reference(q, k, k_eff)
        except RuntimeError:
            return _causal_topk_key_tiled(q, k, k_eff)

    return causal_topk_reference(q, k, k_eff)
