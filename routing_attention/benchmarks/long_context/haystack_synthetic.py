"""Neutral haystack filler for synthetic pointer/address benchmarks."""

from __future__ import annotations

import random

from routing_attention.benchmarks.long_context.haystack import HaystackGenerator


class SyntheticNoiseHaystack(HaystackGenerator):
    """
    Filler using only ``~`` ``^`` ````` and spaces — never ADDR/VAL/ACTIVE tokens.

    Prevents accidental overlap with structured needle records.
    """

    _alphabet = "~^`"

    def generate(self, rng: random.Random, num_chars: int) -> str:
        if num_chars <= 0:
            return ""
        chunk = 8
        parts = [
            "".join(rng.choices(self._alphabet, k=chunk))
            for _ in range(num_chars // (chunk + 1) + 1)
        ]
        text = " ".join(parts)
        if len(text) >= num_chars:
            return text[:num_chars]
        # Pad with the same alphabet — never inject letters/digits that could mimic needles.
        pad_alphabet = self._alphabet + " "
        need = num_chars - len(text)
        return text + "".join(rng.choices(pad_alphabet, k=need))
