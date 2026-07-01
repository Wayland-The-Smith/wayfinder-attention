"""Pluggable haystack generators for long-context retrieval benchmarks."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from pathlib import Path


class HaystackGenerator(ABC):
    """Produce filler text chunks for procedural benchmark samples."""

    @staticmethod
    def _pad_to_exact(text: str, num_chars: int, rng: random.Random | None = None) -> str:
        if num_chars <= 0:
            return ""
        if len(text) >= num_chars:
            return text[:num_chars]
        need = num_chars - len(text)
        if rng is None:
            pad = " " * need
        else:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
            pad = "".join(rng.choices(alphabet, k=need))
        return text + pad

    @abstractmethod
    def generate(self, rng: random.Random, num_chars: int) -> str:
        """Return exactly num_chars of haystack text (space-padded if needed)."""


class RandomTokenHaystack(HaystackGenerator):
    """Random alphanumeric token stream."""

    _alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

    def generate(self, rng: random.Random, num_chars: int) -> str:
        if num_chars <= 0:
            return ""
        # ~6 chars per word avg -> preallocate word count
        n_words = max(1, num_chars // 6)
        words = [
            "".join(rng.choices(self._alphabet, k=rng.randint(3, 10)))
            for _ in range(n_words)
        ]
        return self._pad_to_exact(" ".join(words), num_chars, rng)


class RandomSentenceHaystack(HaystackGenerator):
    """Template natural-language-like sentences."""

    _subjects = (
        "The analyst",
        "A researcher",
        "The system",
        "One module",
        "A process",
        "The model",
        "An operator",
        "The pipeline",
        "A worker",
        "The service",
        "A collector",
        "The indexer",
    )
    _verbs = (
        "observed",
        "processed",
        "recorded",
        "updated",
        "validated",
        "indexed",
        "streamed",
        "buffered",
        "scanned",
        "compiled",
        "merged",
        "sharded",
    )
    _objects = (
        "a large batch of tokens",
        "several context windows",
        "multiple hidden states",
        "routing metadata",
        "attention statistics",
        "sequence fragments",
        "embedding tables",
        "document spans",
        "tokenized paragraphs",
        "cached activations",
        "retrieval candidates",
        "context shards",
    )

    def generate(self, rng: random.Random, num_chars: int) -> str:
        if num_chars <= 0:
            return ""
        avg_len = 72  # typical sentence length in chars
        n_sents = max(1, (num_chars // avg_len) + 1)
        parts = [
            (
                f"{rng.choice(self._subjects)} {rng.choice(self._verbs)} "
                f"{rng.choice(self._objects)} during cycle {rng.randint(1, 9999)} "
                f"with tag {rng.randint(1000000, 9999999)}."
            )
            for _ in range(n_sents)
        ]
        return self._pad_to_exact(" ".join(parts), num_chars, rng)


class StructuredRecordHaystack(HaystackGenerator):
    """Structured key-value records as filler."""

    def generate(self, rng: random.Random, num_chars: int) -> str:
        if num_chars <= 0:
            return ""
        n_recs = max(1, num_chars // 48)
        parts = [
            (
                f"RecordID={rng.randint(1000, 99999)} "
                f"status={rng.choice(['ok', 'pending', 'archived'])} "
                f"value={rng.randint(0, 9999)}."
            )
            for _ in range(n_recs)
        ]
        return self._pad_to_exact(" ".join(parts), num_chars, rng)


class TinyStoriesHaystack(HaystackGenerator):
    """Optional TinyStories text chunks; falls back to RandomSentenceHaystack."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self._fallback = RandomSentenceHaystack()
        self._chunks: list[str] | None = None

    def _load_chunks(self) -> list[str]:
        if self._chunks is not None:
            return self._chunks
        if self.path is None or not self.path.exists():
            self._chunks = []
            return self._chunks
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"\n\s*\n", text)
        self._chunks = [p.strip() for p in parts if len(p.strip()) > 40]
        return self._chunks

    def generate(self, rng: random.Random, num_chars: int) -> str:
        chunks = self._load_chunks()
        if not chunks:
            return self._fallback.generate(rng, num_chars)
        out: list[str] = []
        total = 0
        while total < num_chars:
            chunk = rng.choice(chunks)
            out.append(chunk)
            total += len(chunk) + 1
        return self._pad_to_exact(" ".join(out), num_chars, rng)


HAYSTACK_REGISTRY: dict[str, type[HaystackGenerator]] = {
    "random_tokens": RandomTokenHaystack,
    "random_sentences": RandomSentenceHaystack,
    "structured_records": StructuredRecordHaystack,
    "tinystories": TinyStoriesHaystack,
}


def _register_synthetic_haystack() -> None:
    from routing_attention.benchmarks.long_context.haystack_synthetic import SyntheticNoiseHaystack

    HAYSTACK_REGISTRY["synthetic_noise"] = SyntheticNoiseHaystack


_register_synthetic_haystack()


def build_haystack_generator(mode: str, tinystories_path: str | None = None) -> HaystackGenerator:
    if mode not in HAYSTACK_REGISTRY:
        raise ValueError(f"Unknown haystack mode '{mode}'. Available: {list(HAYSTACK_REGISTRY)}")
    if mode == "tinystories":
        return TinyStoriesHaystack(tinystories_path)
    return HAYSTACK_REGISTRY[mode]()
