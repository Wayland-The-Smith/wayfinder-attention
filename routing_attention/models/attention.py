"""Attention mechanism implementations for comparison and routing experiments."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from routing_attention.models.learned_address import LearnedAddressModule
from routing_attention.models.router import MultiScaleRouter, PerLayerRouter, RouterMLP
from routing_attention.models.fast_attention import (
    fla_causal_linear_attention,
    flex_sliding_window_attention,
    require_fla,
    require_flex_attention,
)
from routing_attention.retrieval.index import RetrievalConfig, RoutingRetriever, retrieval_config_from_dict


def _resolve_retrieval_cfg(
    retrieval_config: RetrievalConfig | dict | None,
    top_k: int,
) -> RetrievalConfig:
    if isinstance(retrieval_config, dict):
        return retrieval_config_from_dict(retrieval_config)
    return retrieval_config or RetrievalConfig(top_k=top_k)


def _use_fused_sparse_path(cfg: RetrievalConfig | None) -> bool:
    return bool(cfg and cfg.use_fused_sparse)


def _sparse_meat_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    candidate_idx: torch.Tensor,
    scale: float,
    dropout: nn.Dropout,
    cfg: RetrievalConfig | None,
) -> torch.Tensor:
    """Meat attention on top-K neighbors (fused Triton or reference gather path)."""
    if _use_fused_sparse_path(cfg):
        from routing_attention.kernels.fused_sparse import sparse_meat_attention

        out = sparse_meat_attention(q, k, v, candidate_idx, scale, method="auto")
    else:
        k_sel, v_sel = _gather_kv_heads(k, v, candidate_idx.long())
        q_exp = q.unsqueeze(3)
        scores = (q_exp * k_sel).sum(-1) * scale
        attn = F.softmax(scores, dim=-1)
        attn = dropout(attn)
        out = (attn.unsqueeze(-1) * v_sel).sum(dim=3)
    return out


def _gather_kv_heads(
    k: torch.Tensor,
    v: torch.Tensor,
    candidate_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather keys/values at candidate indices along the sequence dimension."""
    B, n_heads = k.shape[0], k.shape[1]
    b_idx = torch.arange(B, device=k.device).view(B, 1, 1, 1)
    h_idx = torch.arange(n_heads, device=k.device).view(1, n_heads, 1, 1)
    idx = candidate_idx.unsqueeze(1)
    return k[b_idx, h_idx, idx, :], v[b_idx, h_idx, idx, :]


class DenseAttention(nn.Module):
    """Standard scaled dot-product multi-head attention with optional weight return."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        band_mask: Optional[torch.Tensor] = None,
        return_per_head: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if band_mask is not None:
            scores = scores.masked_fill(~band_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            key_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if return_attention:
            if return_per_head:
                return out, attn  # (B, H, T, T)
            attn_avg = attn.mean(dim=1)
            return out, attn_avg
        return out


class DenseSDPAAttention(DenseAttention):
    """
    Production dense attention via PyTorch SDPA (FlashAttention / mem-efficient when available).

    Same Q/K/V projections as DenseAttention — checkpoint-compatible. Uses is_causal=True
    so the backend never materializes a full T×T score matrix.
    """

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        band_mask: Optional[torch.Tensor] = None,
        return_per_head: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention or band_mask is not None:
            return super().forward(
                x, attn_mask, return_attention, band_mask, return_per_head
            )

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2).expand(B, self.n_heads, T, T)
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            sdpa_mask = sdpa_mask & causal.unsqueeze(0)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=sdpa_mask is None,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """
    Causal linear attention baseline via flash-linear-attention (``fla``).

    ELU+1 feature map; fused chunk kernel (no per-token Python loops).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        require_fla()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        band_mask: Optional[torch.Tensor] = None,
        return_per_head: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = fla_causal_linear_attention(q, k, v, self.scale)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        # Match dense SDPA attention dropout (dropout_p on attn weights); FLA has no
        # weight-level hook, so regularize the aggregated context before out_proj.
        out = self.dropout(out)
        out = self.out_proj(out)

        if return_attention:
            attn_dense = torch.zeros(B, T, T, device=x.device)
            return out, attn_dense
        return out


def _sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scale: float,
    dropout: nn.Dropout,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Causal sliding-window attention in O(T * window) memory (no full T×T matrix)."""
    B, H, T, D = q.shape
    out = torch.zeros_like(v)
    drop_p = dropout.p if dropout.training else 0.0
    for t in range(T):
        start = max(0, t - window_size)
        k_win = k[:, :, start : t + 1, :]
        v_win = v[:, :, start : t + 1, :]
        scores = (q[:, :, t : t + 1, :] @ k_win.transpose(-2, -1)) * scale
        if attn_mask is not None:
            mask = attn_mask[:, start : t + 1].bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if drop_p > 0:
            attn = F.dropout(attn, drop_p, training=True)
        out[:, :, t, :] = (attn @ v_win).squeeze(2)
    return out


class LocalAttention(nn.Module):
    """Sliding-window local attention via PyTorch Flex Attention (compiled)."""

    def __init__(self, d_model: int, n_heads: int, window_size: int = 64, dropout: float = 0.1):
        super().__init__()
        require_flex_attention()
        assert d_model % n_heads == 0
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Production path: Flex Attention (compiled). Procedural NIAH samples have no padding.
        if attn_mask is not None and not attn_mask.all():
            out = _sliding_window_attention(
                q, k, v, self.window_size, self.scale, self.dropout, attn_mask
            )
        else:
            out = flex_sliding_window_attention(
                q, k, v, self.window_size, self.scale, self.dropout
            )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        if return_attention:
            attn_dense = torch.zeros(B, T, T, device=x.device)
            return out, attn_dense
        return out


class KeyVectorSparseAttention(nn.Module):
    """
    kNN-on-key-vectors baseline (literature comparison).

    Retrieves candidates by Q/K dot-product geometry using key vectors,
    not a dedicated routing space.

    Retrieval policy (via ``retrieval_config``):
      - ``apply_to_key_vector=False`` (default): inline exact top-K — for task gates /
        fine-tune comparisons that must match the original brute-force path.
      - ``apply_to_key_vector=True``: ``RoutingRetriever`` (brute / FAISS / HNSW per
        ``method``) — for scaling benchmarks and production long-context paths.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        top_k: int = 32,
        dropout: float = 0.1,
        retrieval_config: RetrievalConfig | dict | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k

        self._retrieval_cfg = _resolve_retrieval_cfg(retrieval_config, top_k)
        self.use_fast_retrieval = self._retrieval_cfg.apply_to_key_vector
        self.retriever = (
            RoutingRetriever(self._retrieval_cfg)
            if self.use_fast_retrieval or self._retrieval_cfg.use_fused_sparse
            else None
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _search_vectors(self, k: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Head-mean Q/K vectors used only for index selection."""
        return q.mean(dim=1), k.mean(dim=1)  # (B, T, head_dim)

    def _retrieve_by_keys_inline(self, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Legacy inline exact top-K (task-gate / reproducibility path)."""
        q_mean, k_mean = self._search_vectors(q, k)
        sim = torch.matmul(q_mean, k_mean.transpose(-2, -1))  # (B, T, T)
        T = sim.shape[-1]
        causal = torch.tril(torch.ones(T, T, device=sim.device, dtype=torch.bool))
        sim = sim.masked_fill(~causal.unsqueeze(0), float("-inf"))
        k_eff = min(self.top_k, T)
        _, indices = torch.topk(sim, k=k_eff, dim=-1)
        return indices

    def _retrieve_by_keys(self, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Top-K index selection — inline or RoutingRetriever depending on config."""
        if not self.use_fast_retrieval:
            return self._retrieve_by_keys_inline(k, q)
        q_mean, k_mean = self._search_vectors(q, k)
        return self.retriever(q_mean, k_mean, top_k=self.top_k)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if _use_fused_sparse_path(self._retrieval_cfg):
            from routing_attention.kernels.fused_sparse import fused_sparse_attention

            q_mean, k_mean = self._search_vectors(q, k)
            meat, candidate_idx = fused_sparse_attention(
                q_mean,
                k_mean,
                q,
                k,
                v,
                self.top_k,
                self.scale,
                retrieval_method=self._retrieval_cfg.method,
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)
        else:
            candidate_idx = self._retrieve_by_keys(k, q)
            meat = _sparse_meat_forward(
                q, k, v, candidate_idx, self.scale, self.dropout, self._retrieval_cfg
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(out)

        if return_attention:
            attn_dense = torch.zeros(B, T, T, device=x.device)
            k_eff = candidate_idx.shape[-1]
            for b in range(B):
                for i in range(T):
                    for ki in range(k_eff):
                        j = candidate_idx[b, i, ki].item()
                        attn_dense[b, i, j] = 1.0 / k_eff
            return out, attn_dense
        return out


class LearnedAddressSparseAttention(nn.Module):
    """
    Sparse attention: learned addresses pick candidates; Q/K/V meat runs on the subset.

    Search index (addresses) is trained with routing loss; meat is trained on task loss.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        top_k: int = 32,
        address_dim: int = 32,
        similarity: Literal["symmetric", "asymmetric"] = "asymmetric",
        dropout: float = 0.1,
        address_module: LearnedAddressModule | None = None,
        retrieval_config: RetrievalConfig | dict | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k

        self._retrieval_cfg = _resolve_retrieval_cfg(retrieval_config, top_k)
        self.retriever = RoutingRetriever(self._retrieval_cfg)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.address = address_module or LearnedAddressModule(
            d_model=d_model,
            address_dim=address_dim,
            similarity=similarity,
        )

    def set_address_module(self, address_module: LearnedAddressModule) -> None:
        self.address = address_module

    def _retrieve_candidates(self, x: torch.Tensor) -> torch.Tensor:
        q_addr, k_addr = self.address.forward_query_key(x)
        return self.retriever(q_addr, k_addr, top_k=self.top_k)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if _use_fused_sparse_path(self._retrieval_cfg):
            from routing_attention.kernels.fused_sparse import fused_sparse_attention

            q_addr, k_addr = self.address.forward_query_key(x)
            meat, candidate_idx = fused_sparse_attention(
                q_addr,
                k_addr,
                q,
                k,
                v,
                self.top_k,
                self.scale,
                retrieval_method=self._retrieval_cfg.method,
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)
        else:
            candidate_idx = self._retrieve_candidates(x)
            meat = _sparse_meat_forward(
                q, k, v, candidate_idx, self.scale, self.dropout, self._retrieval_cfg
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(out)

        if return_attention:
            attn_dense = torch.zeros(B, T, T, device=x.device)
            k_eff = candidate_idx.shape[-1]
            for b in range(B):
                for i in range(T):
                    for ki in range(k_eff):
                        j = candidate_idx[b, i, ki].item()
                        attn_dense[b, i, j] = 1.0 / k_eff
            return out, attn_dense
        return out


class RoutingSparseAttention(nn.Module):
    """
    Sparse attention using routing vectors for candidate retrieval.

    Supports per-layer RouterMLP, PerLayerRouter (via layer_idx), or MultiScaleRouter.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        router: nn.Module,
        top_k: int = 32,
        dropout: float = 0.1,
        layer_idx: int = 0,
        retrieval_mode: Literal["routing", "key_vector"] = "routing",
        retrieval_config: RetrievalConfig | dict | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k
        self.router = router
        self.layer_idx = layer_idx
        self.retrieval_mode = retrieval_mode

        self._retrieval_cfg = _resolve_retrieval_cfg(retrieval_config, top_k)
        self.retriever = RoutingRetriever(self._retrieval_cfg)
        self._last_retrieval_ms: float = 0.0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if retrieval_mode == "key_vector":
            self._key_baseline = KeyVectorSparseAttention(
                d_model, n_heads, top_k, dropout, retrieval_config=retrieval_config
            )

    def _get_query_key_routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized query/key routing vectors for retrieval."""
        if isinstance(self.router, PerLayerRouter):
            router = self.router.get_router(self.layer_idx)
            return router.forward_query_key(x)
        if hasattr(self.router, "forward_query_key"):
            return self.router.forward_query_key(x)
        r = self.router(x)
        return r, r

    def _retrieve_candidates(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient vector search: top-k keys per query by routing similarity.

        Uses RoutingRetriever (brute GEMM or FAISS ANN depending on config/seq length).
        """
        if isinstance(self.router, MultiScaleRouter):
            return self.router.retrieve_indices(x, self.top_k, self.layer_idx)

        q, k = self._get_query_key_routing(x)
        return self.retriever(q, k, top_k=self.top_k)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        precomputed_routing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.retrieval_mode == "key_vector":
            return self._key_baseline(x, attn_mask=attn_mask, return_attention=return_attention)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if _use_fused_sparse_path(self._retrieval_cfg):
            from routing_attention.kernels.fused_sparse import fused_sparse_attention

            q_r, k_r = self._get_query_key_routing(x)
            meat, candidate_idx = fused_sparse_attention(
                q_r,
                k_r,
                q,
                k,
                v,
                self.top_k,
                self.scale,
                retrieval_method=self._retrieval_cfg.method,
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)
        else:
            candidate_idx = self._retrieve_candidates(x)
            meat = _sparse_meat_forward(
                q, k, v, candidate_idx, self.scale, self.dropout, self._retrieval_cfg
            )
            out = meat.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(out)

        if return_attention:
            attn_dense = torch.zeros(B, T, T, device=x.device)
            k_eff = candidate_idx.shape[-1]
            for b in range(B):
                for i in range(T):
                    for ki in range(k_eff):
                        j = candidate_idx[b, i, ki].item()
                        attn_dense[b, i, j] = 1.0 / k_eff
            return out, attn_dense
        return out
