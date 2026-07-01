"""Per-token learned address vectors for routing (decoupled from Q/K attention meat)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAddressModule(nn.Module):
    """
    Maps hidden states to low-dimensional query/key *addresses* used only for retrieval.

    Attention meat (q_proj/k_proj/v_proj) stays separate and is trained on the task objective.
    Addresses are trained with routing loss (InfoNCE, etc.) in Phase C.
    """

    def __init__(
        self,
        d_model: int,
        address_dim: int = 32,
        similarity: Literal["symmetric", "asymmetric"] = "asymmetric",
        normalize: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.address_dim = address_dim
        self.similarity = similarity
        self.normalize = normalize

        self.q_addr_proj = nn.Linear(d_model, address_dim)
        if similarity == "asymmetric":
            self.k_addr_proj = nn.Linear(d_model, address_dim)
        else:
            self.k_addr_proj = None

    def forward_query_key(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (query_address, key_address) with shape (B, T, address_dim)."""
        q = self.q_addr_proj(hidden)
        k = self.k_addr_proj(hidden) if self.k_addr_proj is not None else q
        if self.normalize:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        return q, k

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Symmetric address (for RouterTrainer / loss helpers expecting forward())."""
        q, _ = self.forward_query_key(hidden)
        return q

    def retrieval_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pairwise address similarity (B, T, T) for Recall@K evaluation."""
        q, k = self.forward_query_key(hidden)
        return torch.matmul(q, k.transpose(-2, -1))


class PerLayerAddressBook(nn.Module):
    """Independent learned address module per transformer layer."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        address_dim: int = 32,
        similarity: Literal["symmetric", "asymmetric"] = "asymmetric",
        normalize: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.addresses = nn.ModuleList([
            LearnedAddressModule(
                d_model=d_model,
                address_dim=address_dim,
                similarity=similarity,
                normalize=normalize,
            )
            for _ in range(n_layers)
        ])

    def get_address(self, layer_idx: int) -> LearnedAddressModule:
        return self.addresses[layer_idx]

    def get_router(self, layer_idx: int) -> LearnedAddressModule:
        """Alias for RouterTrainer / per-layer training compatibility."""
        return self.get_address(layer_idx)

    def retrieval_scores(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.get_address(layer_idx).retrieval_scores(hidden)


def build_address_book_from_config(config: dict) -> PerLayerAddressBook:
    """Build per-layer address book from config (learned_address + model sections)."""
    la_cfg = config.get("learned_address", {})
    model_cfg = config["model"]
    return PerLayerAddressBook(
        n_layers=model_cfg["n_layers"],
        d_model=model_cfg["d_model"],
        address_dim=la_cfg.get("address_dim", la_cfg.get("routing_dim", 32)),
        similarity=la_cfg.get("similarity", "asymmetric"),
        normalize=la_cfg.get("normalize", True),
    )


def attach_address_book_to_model(model: nn.Module, address_book: PerLayerAddressBook) -> None:
    """Attach address book to a TransformerLM (used after loading dense checkpoint)."""
    model.address_book = address_book
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        if hasattr(attn, "set_address_module"):
            attn.set_address_module(address_book.get_address(layer_idx))


def ensure_address_book_on_model(
    model: nn.Module,
    config: dict,
    device: torch.device | str | None = None,
) -> PerLayerAddressBook:
    """Create and attach address book if the model does not already have one."""
    if getattr(model, "address_book", None) is not None:
        book = model.address_book
        if device is not None:
            book = book.to(device)
            attach_address_book_to_model(model, book)
        else:
            attach_address_book_to_model(model, book)
        return book
    book = build_address_book_from_config(config)
    if device is not None:
        book = book.to(device)
    model.address_book = book
    attach_address_book_to_model(model, book)
    return book
