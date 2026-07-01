"""
Efficient routing-vector retrieval for sparse attention.

Transforms naive O(T²) all-pairs search into hardware-friendly paths:

  fused_causal  — Triton fused causal top-K (no T×T materialization; preferred on CUDA)
  brute_force   — cuBLAS batched GEMM + topk (materializes T×T scores)
  faiss_flat    — FAISS IndexFlatIP (legacy; slow when index rebuilt per forward)
  faiss_hnsw    — FAISS HNSW approximate
  auto          — fused_causal on CUDA when available, else brute_force

Causal masking: retrieve k * oversample candidates, filter j ≤ i, keep top-k valid.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import faiss

    _HAS_FAISS = True
except (ImportError, AttributeError):
    _HAS_FAISS = False


def _tensor_to_numpy_f32(tensor: torch.Tensor) -> "np.ndarray":
    """
    float32 numpy array from a tensor.

    Uses ctypes memcpy when torch.Tensor.numpy() is broken (common on Windows
    after conda faiss-gpu pulls a mismatched numpy build).
    """
    if np is None:
        raise RuntimeError("numpy is required for FAISS retrieval")
    t = tensor.detach().float().cpu().contiguous()
    try:
        return t.numpy()
    except RuntimeError:
        arr = np.empty(t.shape, dtype=np.float32)
        if t.numel() == 0:
            return arr
        ctypes.memmove(
            arr.ctypes.data,
            ctypes.c_void_p(t.data_ptr()),
            t.numel() * arr.itemsize,
        )
        return arr


def numpy_bridge_available() -> bool:
    """True if torch tensors can be converted to numpy (needed for FAISS CPU path)."""
    if np is None:
        return False
    try:
        _ = _tensor_to_numpy_f32(torch.zeros(1))
        return True
    except Exception:
        return False


RetrievalMethod = Literal[
    "fused_causal",
    "brute_force",
    "faiss_flat",
    "faiss_hnsw",
    "auto",
]


@dataclass
class RetrievalConfig:
    method: RetrievalMethod = "auto"
    top_k: int = 32
    max_seq_len: int = 8192
    oversample: int = 4          # retrieve oversample*k then causal-filter
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    # auto-selection thresholds
    brute_force_max_seq: int = 2048
    faiss_flat_max_seq: int = 16384
    hnsw_m: int = 32             # HNSW connectivity
    hnsw_ef_search: int = 64
    use_gpu: bool = True
    # Key-vector baseline uses inline top-K unless explicitly enabled (task gates vs scaling).
    apply_to_key_vector: bool = False
    # Layer 3: Triton fused retrieval + sparse meat attention (single pipeline).
    use_fused_sparse: bool = False


class RoutingRetriever(nn.Module):
    """
    Causal top-k retrieval in routing vector space.

    Query i retrieves keys j ≤ i with highest q_i · k_j (cosine if normalized).
    """

    def __init__(self, config: RetrievalConfig | None = None):
        super().__init__()
        self.config = config or RetrievalConfig()
        # Cached causal allow-mask: True where j <= i
        causal = torch.tril(torch.ones(self.config.max_seq_len, self.config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_allow", causal, persistent=False)

    def ensure_capacity(self, seq_len: int) -> None:
        """Grow cached causal mask / config when benchmarking longer sequences."""
        seq_len = int(seq_len)
        if seq_len > self.config.max_seq_len:
            self.config.max_seq_len = seq_len
        if seq_len > self.causal_allow.shape[0]:
            device = self.causal_allow.device
            causal = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            self.register_buffer("causal_allow", causal, persistent=False)

    def resolve_method(self, seq_len: int) -> RetrievalMethod:
        cfg = self.config
        if cfg.method != "auto":
            return cfg.method
        from routing_attention.kernels.causal_topk import causal_topk_available

        if causal_topk_available():
            return "fused_causal"
        if seq_len <= cfg.brute_force_max_seq:
            return "brute_force"
        if seq_len <= cfg.faiss_flat_max_seq:
            return "faiss_flat" if _HAS_FAISS else "brute_force"
        return "faiss_hnsw" if _HAS_FAISS else "brute_force"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T, R) query routing vectors
            key:   (B, T, R) key routing vectors
        Returns:
            indices: (B, T, k) key positions per query (causal)
        """
        k = top_k or self.config.top_k
        if query.shape != key.shape:
            raise ValueError(
                f"Retrieval query/key shape mismatch: query={tuple(query.shape)}, "
                f"key={tuple(key.shape)}. Ensure model.max_seq_len and "
                f"retrieval.max_seq_len cover the benchmark sequence length."
            )
        T = query.shape[1]
        self.ensure_capacity(T)
        method = method or self.resolve_method(T)

        if method == "fused_causal":
            return self._retrieve_fused_causal(query, key, k)
        if method == "brute_force":
            return self._retrieve_brute_force(query, key, k)
        if method == "faiss_flat":
            try:
                return self._retrieve_faiss(query, key, k, approximate=False)
            except Exception:
                return self._retrieve_brute_force(query, key, k)
        if method == "faiss_hnsw":
            try:
                return self._retrieve_faiss(query, key, k, approximate=True)
            except Exception:
                return self._retrieve_brute_force(query, key, k)
        raise ValueError(f"Unknown retrieval method: {method}")

    def _causal_allow(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask j <= i; grows on demand when seq_len exceeds cached buffer."""
        if seq_len <= self.causal_allow.shape[0]:
            return self.causal_allow[:seq_len, :seq_len]
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.dtype == "float16":
            return x.half()
        if self.config.dtype == "bfloat16":
            return x.bfloat16()
        return x.float()

    def _retrieve_fused_causal(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Triton fused causal top-K — no full (T,T) score matrix."""
        from routing_attention.kernels.causal_topk import causal_topk

        q = self._cast(query)
        k = self._cast(key)
        indices = causal_topk(q, k, top_k, method="fused_causal", dtype=self.config.dtype)
        return indices.long()

    def _retrieve_brute_force(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Fused batched GEMM + causal topk — exact, GPU-friendly for moderate T."""
        B, T, R = query.shape
        q = self._cast(query)
        k = self._cast(key)

        # (B, T, T) inner products — single cuBLAS batched matmul
        scores = torch.bmm(q, k.transpose(1, 2))

        allow = self._causal_allow(T, query.device).unsqueeze(0)
        scores = scores.masked_fill(~allow, float("-inf"))

        k_eff = min(top_k, T)
        _, indices = torch.topk(scores, k=k_eff, dim=-1, sorted=False)
        return indices

    def _retrieve_faiss(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int,
        approximate: bool = False,
    ) -> torch.Tensor:
        """
        FAISS inner-product search with causal post-filtering.

        Uses oversampling because ANN may return future positions (j > i).
        """
        if not _HAS_FAISS:
            return self._retrieve_brute_force(query, key, top_k)

        B, T, R = query.shape
        k_search = min(self.config.oversample * top_k, T)
        device = query.device
        use_gpu = self.config.use_gpu and device.type == "cuda" and hasattr(faiss, "StandardGpuResources")

        all_indices = torch.zeros(B, T, top_k, dtype=torch.long, device=device)

        q_cpu = _tensor_to_numpy_f32(query)
        k_cpu = _tensor_to_numpy_f32(key)

        for b in range(B):
            keys = k_cpu[b]
            queries = q_cpu[b]

            if approximate:
                index = faiss.IndexHNSWFlat(R, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efSearch = self.config.hnsw_ef_search
            else:
                index = faiss.IndexFlatIP(R)

            if use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            index.add(keys)
            _, nn_idx = index.search(queries, k_search)

            # Causal filter per query row
            filtered = torch.full((T, top_k), 0, dtype=torch.long, device=device)
            for i in range(T):
                valid = [int(j) for j in nn_idx[i] if 0 <= j <= i]
                if len(valid) < top_k:
                    # Fallback: exact topk on prefix for this row
                    q_row = query[b, i : i + 1]
                    k_prefix = key[b, : i + 1]
                    sim = torch.matmul(q_row, k_prefix.transpose(0, 1))
                    _, fb = torch.topk(sim, k=min(top_k, i + 1), dim=-1)
                    valid = fb[0].tolist()
                for ki in range(top_k):
                    filtered[i, ki] = valid[ki % len(valid)]
            all_indices[b] = filtered

        return all_indices

    @torch.no_grad()
    def benchmark(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        top_k: int | None = None,
        methods: list[RetrievalMethod] | None = None,
        num_runs: int = 20,
    ) -> dict[str, float]:
        """Time each retrieval method in milliseconds."""
        top_k = top_k or self.config.top_k
        T = query.shape[1]
        if methods is None:
            resolved = self.resolve_method(T)
            methods = [resolved]
            if resolved not in ("brute_force",):
                methods = ["brute_force", resolved]
            from routing_attention.kernels.causal_topk import causal_topk_available

            if causal_topk_available() and resolved != "fused_causal":
                methods.append("fused_causal")

        results = {}
        for m in methods:
            if m in ("faiss_flat", "faiss_hnsw") and not _HAS_FAISS:
                continue
            if m in ("faiss_flat", "faiss_hnsw") and not numpy_bridge_available():
                continue
            # warmup
            for _ in range(3):
                self.forward(query, key, top_k, method=m)
            if query.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_runs):
                self.forward(query, key, top_k, method=m)
            if query.device.type == "cuda":
                torch.cuda.synchronize()
            results[m] = (time.perf_counter() - t0) / num_runs * 1000
        return results


def benchmark_retrieval(
    seq_len: int,
    batch_size: int = 1,
    routing_dim: int = 32,
    top_k: int = 32,
    device: str = "cuda",
) -> dict:
    """Standalone retrieval benchmark utility."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, seq_len, routing_dim, device=dev)
    k = torch.randn(batch_size, seq_len, routing_dim, device=dev)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    retriever = RoutingRetriever(RetrievalConfig(top_k=top_k, max_seq_len=seq_len)).to(dev)
    times = retriever.benchmark(q, k)
    return {"seq_len": seq_len, "batch_size": batch_size, "retrieval_ms": times}


def sync_retrieval_config_dict(cfg: dict | None, model_max_seq_len: int) -> dict:
    """Ensure retrieval.max_seq_len covers the model / benchmark sequence length."""
    out = dict(cfg or {})
    out["max_seq_len"] = max(int(out.get("max_seq_len", 8192)), int(model_max_seq_len))
    return out


def patch_model_retrievers(model: torch.nn.Module, max_seq_len: int) -> None:
    """Resize all RoutingRetriever modules on a TransformerLM before long-seq benchmarks."""
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        candidates = [attn, getattr(attn, "retriever", None)]
        baseline = getattr(attn, "_key_baseline", None)
        if baseline is not None:
            candidates.append(baseline)
            candidates.append(getattr(baseline, "retriever", None))
        for mod in candidates:
            if mod is not None and hasattr(mod, "ensure_capacity"):
                mod.ensure_capacity(max_seq_len)


def retrieval_config_from_dict(cfg: dict) -> RetrievalConfig:
    """Build RetrievalConfig from YAML dict."""
    return RetrievalConfig(
        method=cfg.get("method", "auto"),
        top_k=cfg.get("top_k", 32),
        max_seq_len=cfg.get("max_seq_len", 8192),
        oversample=cfg.get("oversample", 4),
        dtype=cfg.get("dtype", "float16"),
        brute_force_max_seq=cfg.get("brute_force_max_seq", 2048),
        faiss_flat_max_seq=cfg.get("faiss_flat_max_seq", 16384),
        hnsw_m=cfg.get("hnsw_m", 32),
        hnsw_ef_search=cfg.get("hnsw_ef_search", 64),
        use_gpu=cfg.get("use_gpu", True),
        apply_to_key_vector=cfg.get("apply_to_key_vector", False),
        use_fused_sparse=cfg.get("use_fused_sparse", False),
    )
