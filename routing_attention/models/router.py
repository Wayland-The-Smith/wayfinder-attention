"""Routing projection modules for attention neighborhood prediction."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterMLP(nn.Module):
    """
    Maps hidden states to routing vectors.

    symmetric: one vector r_i, similarity r_i · r_j (chat's default factorization)
    asymmetric: separate query/key vectors, similarity q_i · k_j (closer to Q·K^T)
    """

    def __init__(
        self,
        input_dim: int,
        routing_dim: int = 32,
        hidden_dim: int | None = None,
        normalize: bool = True,
        dropout: float = 0.0,
        use_affine_dot: bool = False,
        similarity: Literal["symmetric", "asymmetric"] = "symmetric",
        n_heads: int = 1,
        per_head_routing: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(routing_dim * 2, input_dim // 4)
        self.normalize = normalize
        self.use_affine_dot = use_affine_dot
        self.similarity = similarity
        self.n_heads = n_heads
        self.per_head_routing = per_head_routing
        self.routing_dim = routing_dim

        out_dim = routing_dim * n_heads if per_head_routing else routing_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        if similarity == "asymmetric":
            self.key_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        if use_affine_dot:
            self.dot_scale = nn.Parameter(torch.tensor(1.0))
            self.dot_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, as_query: bool = True) -> torch.Tensor:
        """
        Returns routing vectors for query (default) or key role.

        Shape: (B, T, R) or (B, T, H, R) if per_head_routing.
        """
        if self.similarity == "asymmetric" and not as_query:
            r = self.key_net(x)
        else:
            r = self.net(x)

        if self.per_head_routing:
            B, T, _ = r.shape
            r = r.view(B, T, self.n_heads, self.routing_dim)

        if self.normalize:
            r = F.normalize(r, dim=-1)
        return r

    def forward_query_key(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (query_routing, key_routing) for asymmetric similarity."""
        q = self.forward(x, as_query=True)
        if self.similarity == "asymmetric":
            k = self.forward(x, as_query=False)
        else:
            k = q
        return q, k

    def pairwise_scores(
        self,
        routing: torch.Tensor,
        key_routing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pairwise scores for loss. routing is query side."""
        if key_routing is None:
            key_routing = routing
        if routing.dim() == 4:  # per-head: (B, T, H, R)
            # Average scores across heads for supervision loss
            scores = torch.einsum("bthq,bshq->bhts", routing, key_routing).mean(dim=2)
        else:
            scores = torch.matmul(routing, key_routing.transpose(-2, -1))
        if self.use_affine_dot:
            scores = scores * self.dot_scale + self.dot_bias
        return scores

    def retrieval_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scores for vector-search retrieval: (B, T, T) where [i,j] = sim(query_i, key_j).
        """
        q, k = self.forward_query_key(x)
        if q.dim() == 4:
            # Per-head retrieval, union handled in attention module; return head-mean scores
            scores = torch.einsum("bthq,bshq->bts", q, k) / q.shape[2]
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))
        return scores


class PerLayerRouter(nn.Module):
    """Independent router per transformer layer."""

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        routing_dim: int = 32,
        hidden_dim: int | None = None,
        normalize: bool = True,
        use_affine_dot: bool = False,
        similarity: str = "symmetric",
        n_heads: int = 1,
        per_head_routing: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.routers = nn.ModuleList([
            RouterMLP(
                input_dim, routing_dim, hidden_dim, normalize,
                use_affine_dot=use_affine_dot,
                similarity=similarity,
                n_heads=n_heads,
                per_head_routing=per_head_routing,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, layer_idx: int, as_query: bool = True) -> torch.Tensor:
        return self.routers[layer_idx](x, as_query=as_query)

    def forward_query_key(self, x: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routers[layer_idx].forward_query_key(x)

    def retrieval_scores(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.routers[layer_idx].retrieval_scores(x)

    def get_router(self, layer_idx: int) -> RouterMLP:
        return self.routers[layer_idx]


class MultiScaleRouter(nn.Module):
    """Local + global routers with merged candidate retrieval."""

    def __init__(
        self,
        input_dim: int,
        routing_dim: int = 32,
        hidden_dim: int | None = None,
        local_window: int = 16,
        similarity: str = "symmetric",
        n_heads: int = 1,
        per_head_routing: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(routing_dim * 2, input_dim // 4)
        self.local_window = local_window
        kw = dict(similarity=similarity, n_heads=n_heads, per_head_routing=per_head_routing)
        self.local_router = RouterMLP(input_dim, routing_dim, hidden_dim, normalize=True, **kw)
        self.global_router = RouterMLP(input_dim, routing_dim, hidden_dim, normalize=True, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.local_router(x), self.global_router(x)], dim=-1)

    def _topk_from_router(self, router: RouterMLP, x: torch.Tensor, k_eff: int) -> torch.Tensor:
        sim = router.retrieval_scores(x)
        T = sim.shape[-1]
        causal = torch.tril(torch.ones(T, T, device=x.device))
        sim = sim.masked_fill(causal.unsqueeze(0) == 0, float("-inf"))
        _, idx = torch.topk(sim, k=min(k_eff, T), dim=-1)
        return idx

    def retrieve_indices(self, x: torch.Tensor, k: int, layer_idx: int = 0) -> torch.Tensor:
        del layer_idx
        B, T, _ = x.shape
        k_half = max(1, k // 2)
        idx_local = self._topk_from_router(self.local_router, x, k_half)
        idx_global = self._topk_from_router(self.global_router, x, k - k_half)
        merged = torch.zeros(B, T, k, dtype=torch.long, device=x.device)
        for b in range(B):
            for i in range(T):
                candidates = torch.cat([idx_local[b, i], idx_global[b, i]]).unique()
                n = min(k, candidates.numel())
                merged[b, i, :n] = candidates[:n]
                if n < k:
                    merged[b, i, n:] = candidates[0]
        return merged

    def routing_for_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.local_router(x), self.global_router(x)


def build_router_from_config(config: dict) -> nn.Module:
    """Factory: single RouterMLP, PerLayerRouter, or MultiScaleRouter."""
    router_cfg = config.get("router", {})
    model_cfg = config.get("model", {})
    mode: Literal["single", "per_layer", "multi_scale"] = router_cfg.get("mode", "per_layer")
    input_dim = model_cfg["d_model"]
    routing_dim = router_cfg.get("routing_dim", 32)
    hidden_dim = router_cfg.get("hidden_dim")
    loss_type = router_cfg.get("loss_type", "infonce")
    normalize = loss_type not in ("mse",)
    use_affine_dot = loss_type == "mse"
    similarity = router_cfg.get("similarity", "symmetric")
    n_heads = model_cfg.get("n_heads", 4)
    per_head_routing = router_cfg.get("per_head_routing", False)

    common = dict(
        similarity=similarity,
        n_heads=n_heads,
        per_head_routing=per_head_routing,
    )

    if mode == "multi_scale":
        return MultiScaleRouter(
            input_dim=input_dim,
            routing_dim=routing_dim,
            hidden_dim=hidden_dim,
            local_window=router_cfg.get("local_window", 16),
            **common,
        )
    if mode == "per_layer":
        return PerLayerRouter(
            n_layers=model_cfg["n_layers"],
            input_dim=input_dim,
            routing_dim=routing_dim,
            hidden_dim=hidden_dim,
            normalize=normalize,
            use_affine_dot=use_affine_dot,
            **common,
        )
    return RouterMLP(
        input_dim=input_dim,
        routing_dim=routing_dim,
        hidden_dim=hidden_dim,
        normalize=normalize,
        use_affine_dot=use_affine_dot,
        **common,
    )
