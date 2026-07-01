"""Character tokenizer for procedural long-context benchmark text."""

from __future__ import annotations

import numpy as np


class BenchmarkTokenizer:
    """Maps printable ASCII to token ids in [1, vocab_size-1]; pad=0. One char -> one token."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self._mod = vocab_size - 1
        self._answer_prefix_ids: list[int] | None = None
        self._question_prefix_ids: list[int] | None = None

    def encode(self, text: str) -> list[int]:
        return self.encode_array(text).tolist()

    def encode_array(self, text: str) -> np.ndarray:
        """Vectorized encode — exactly one token per input character."""
        n = len(text)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        if text.isascii():
            codes = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        else:
            codes = np.fromiter((ord(c) for c in text), dtype=np.int32, count=n)
            return ((codes % self._mod) + 1).astype(np.int32, copy=False)
        return ((codes.astype(np.int32) % self._mod) + 1)

    def answer_prefix_ids(self, answer_prefix: str) -> list[int]:
        if self._answer_prefix_ids is None:
            self._answer_prefix_ids = self.encode(answer_prefix)
        return self._answer_prefix_ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        chars: list[str] = []
        for tid in ids:
            if tid <= 0:
                continue
            chars.append(chr((int(tid) - 1) % 256))
        return "".join(chars)

    def normalize_answer(self, text: str) -> str:
        return " ".join(text.strip().lower().split())
