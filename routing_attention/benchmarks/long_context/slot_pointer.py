"""
Token-native slot-pointer benchmark (Experiment 7).

Sequence layout (length T, default 2048):
  - Indices [0, T-2]: hay tokens and contiguous quads ``[addr][;][value][,]``
  - Index T-1: question token (= addr token id being queried)
  - Label: index of the value token for the matching quad

Vocabulary (129 ids: 0=pad, 1..128 content):
  - 1..100   semantic pool (addr and value drawn from same set, must differ within quad)
  - 101      semicolon
  - 102      comma
  - 103..128 hay / distractor tokens (26 types)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

PAD_TOKEN_ID = 0
SEMANTIC_TOKEN_START = 1
SEMANTIC_TOKEN_END = 100
TOK_SEMICOLON = 101
TOK_COMMA = 102
HAY_TOKEN_START = 103
HAY_TOKEN_END = 128
SLOT_POINTER_VOCAB_SIZE = 129

SEMANTIC_TOKENS = tuple(range(SEMANTIC_TOKEN_START, SEMANTIC_TOKEN_END + 1))
HAY_TOKENS = tuple(range(HAY_TOKEN_START, HAY_TOKEN_END + 1))
DELIMITER_TOKENS = (TOK_SEMICOLON, TOK_COMMA)


@dataclass(frozen=True)
class SlotQuad:
    """One contiguous addr;value, record in the sequence."""

    addr_token: int
    value_token: int
    start_index: int

    @property
    def value_index(self) -> int:
        return self.start_index + 2

    @property
    def end_index_exclusive(self) -> int:
        return self.start_index + 4

    def write_into(self, tokens: list[int]) -> None:
        tokens[self.start_index] = self.addr_token
        tokens[self.start_index + 1] = TOK_SEMICOLON
        tokens[self.start_index + 2] = self.value_token
        tokens[self.start_index + 3] = TOK_COMMA


@dataclass(frozen=True)
class SlotPointerSample:
    """Fully specified slot-pointer instance before tensor packaging."""

    tokens: tuple[int, ...]
    context_length: int
    question_index: int
    query_addr_token: int
    target_value_index: int
    target_value_token: int
    quads: tuple[SlotQuad, ...]

    def verify(self) -> None:
        trace_value_index(self.tokens, self.question_index)


def _sample_pairs(
    rng: random.Random,
    count: int,
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    if count > len(SEMANTIC_TOKENS):
        raise ValueError(f"need {count} pairs but only {len(SEMANTIC_TOKENS)} semantic tokens")
    addrs = rng.sample(SEMANTIC_TOKENS, count)
    pairs: list[tuple[int, int]] = []
    used_values: set[int] = set()
    for addr in addrs:
        value_pool = [t for t in SEMANTIC_TOKENS if t != addr and t not in used_values]
        if not value_pool:
            raise ValueError("cannot sample distinct value tokens for all pairs")
        value = rng.choice(value_pool)
        used_values.add(value)
        pairs.append((addr, value))
    return pairs


def _place_non_overlapping_quads(
    rng: random.Random,
    *,
    content_length: int,
    num_quads: int,
    max_attempts: int = 512,
) -> list[int] | None:
    """Return quad start indices (length 4 each) in ``[0, content_length)``."""
    if num_quads <= 0:
        return []
    if 4 * num_quads > content_length:
        return None
    for _ in range(max_attempts):
        starts: list[int] = []
        occupied: set[int] = set()
        order = list(range(num_quads))
        rng.shuffle(order)
        ok = True
        for _idx in order:
            if content_length < 4:
                ok = False
                break
            hi = content_length - 4
            placed = False
            for _try in range(64):
                start = rng.randint(0, hi)
                span = set(range(start, start + 4))
                if not span & occupied:
                    starts.append(start)
                    occupied |= span
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and len(starts) == num_quads:
            return starts
    return None


def _place_fixed_grid_quads(
    *,
    content_length: int,
    num_quads: int,
) -> list[int]:
    """Evenly spaced non-overlapping quad starts across ``[0, content_length)``."""
    if num_quads <= 0:
        return []
    if 4 * num_quads > content_length:
        raise ValueError(
            f"cannot place {num_quads} quads in content_length={content_length}"
        )
    gap_total = content_length - 4 * num_quads
    gap = gap_total // (num_quads + 1)
    starts: list[int] = []
    pos = gap
    for _ in range(num_quads):
        if pos + 4 > content_length:
            raise ValueError("fixed grid placement overflow")
        starts.append(pos)
        pos += 4 + gap
    return starts


def _place_quads(
    rng: random.Random,
    *,
    content_length: int,
    num_quads: int,
    placement: str = "random",
    max_attempts: int = 512,
) -> list[int]:
    if placement == "fixed_grid":
        return _place_fixed_grid_quads(content_length=content_length, num_quads=num_quads)
    if placement != "random":
        raise ValueError(f"unknown slot quad placement {placement!r}")
    starts = _place_non_overlapping_quads(
        rng,
        content_length=content_length,
        num_quads=num_quads,
        max_attempts=max_attempts,
    )
    if starts is None:
        raise ValueError(
            f"failed to place {num_quads} quads in content_length={content_length}"
        )
    return starts


def verify_unique_semantic_usage(
    tokens: Sequence[int],
    *,
    question_index: int,
) -> None:
    """Semantic ids may appear only inside quads; all other content positions are hay tokens."""
    q_idx = question_index
    quad_spans: set[int] = set()
    quad_addrs: set[int] = set()
    i = 0
    while i + 3 < q_idx:
        if (
            tokens[i] in SEMANTIC_TOKENS
            and tokens[i + 1] == TOK_SEMICOLON
            and tokens[i + 2] in SEMANTIC_TOKENS
            and tokens[i + 3] == TOK_COMMA
        ):
            if tokens[i] == tokens[i + 2]:
                raise ValueError(f"addr and value identical within quad at {i}")
            quad_addrs.add(tokens[i])
            for pos in range(i, i + 4):
                quad_spans.add(pos)
            i += 4
            continue
        i += 1

    for pos in range(q_idx):
        if pos in quad_spans:
            continue
        if tokens[pos] not in HAY_TOKENS:
            raise ValueError(f"unexpected token {tokens[pos]} at content position {pos}")

    if tokens[q_idx] not in SEMANTIC_TOKENS:
        raise ValueError("question slot must hold a semantic addr token")
    if tokens[q_idx] not in quad_addrs:
        raise ValueError("question addr token must match a quad addr token")


def generate_slot_pointer_tokens(
    rng: random.Random,
    *,
    context_length: int = 2048,
    num_quads: int = 50,
    placement: str = "random",
    enforce_unique_semantics: bool = False,
    max_placement_attempts: int = 512,
) -> SlotPointerSample:
    """
    Build one slot-pointer sequence.

    Raises ``ValueError`` if placement fails (caller should retry with a new seed).
    """
    if context_length < 16:
        raise ValueError("context_length must be >= 16")
    question_index = context_length - 1
    content_length = context_length - 1

    pairs = _sample_pairs(rng, num_quads)
    starts = _place_quads(
        rng,
        content_length=content_length,
        num_quads=num_quads,
        placement=placement,
        max_attempts=max_placement_attempts,
    )

    query_pair_idx = rng.randrange(num_quads)
    query_addr, query_value = pairs[query_pair_idx]

    tokens = [0] * context_length
    quads: list[SlotQuad] = []
    for (addr, value), start in zip(pairs, starts):
        quad = SlotQuad(addr_token=addr, value_token=value, start_index=start)
        quad.write_into(tokens)
        quads.append(quad)

    empty_slots = [i for i in range(content_length) if tokens[i] == 0]
    for i in empty_slots:
        tokens[i] = rng.choice(HAY_TOKENS)

    tokens[question_index] = query_addr

    sample = SlotPointerSample(
        tokens=tuple(tokens),
        context_length=context_length,
        question_index=question_index,
        query_addr_token=query_addr,
        target_value_index=-1,
        target_value_token=query_value,
        quads=tuple(quads),
    )
    target = trace_value_index(sample.tokens, sample.question_index)
    if enforce_unique_semantics:
        verify_unique_semantic_usage(sample.tokens, question_index=sample.question_index)
    return SlotPointerSample(
        tokens=sample.tokens,
        context_length=sample.context_length,
        question_index=sample.question_index,
        query_addr_token=sample.query_addr_token,
        target_value_index=target,
        target_value_token=query_value,
        quads=sample.quads,
    )


def trace_value_index(
    tokens: Sequence[int],
    question_index: int | None = None,
) -> int:
    """
    Return the index of the value token for the quad whose addr matches the question.

    The question token at ``question_index`` (default: last index) is the queried addr id.
    """
    if not tokens:
        raise ValueError("empty token sequence")
    q_idx = len(tokens) - 1 if question_index is None else question_index
    if q_idx < 0 or q_idx >= len(tokens):
        raise ValueError(f"invalid question_index={question_index}")
    query_addr = tokens[q_idx]
    if query_addr not in SEMANTIC_TOKENS:
        raise ValueError(f"question token {query_addr} is not a semantic token")

    matches: list[int] = []
    limit = q_idx
    i = 0
    while i + 3 < limit:
        if (
            tokens[i] == query_addr
            and tokens[i + 1] == TOK_SEMICOLON
            and tokens[i + 2] in SEMANTIC_TOKENS
            and tokens[i + 2] != query_addr
            and tokens[i + 3] == TOK_COMMA
        ):
            matches.append(i + 2)
        i += 1

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one matching quad for addr={query_addr}, found {len(matches)}"
        )
    return matches[0]


def trace_addr_index(tokens: Sequence[int], question_index: int | None = None) -> int:
    """Auxiliary: index of addr token in the matching quad."""
    value_index = trace_value_index(tokens, question_index)
    return value_index - 2


def slot_pointer_value_slot_labels(
    quads: Sequence[SlotQuad],
    target_value_index: int,
) -> tuple[list[int], int]:
    """Return ordered value-token indices and the slot index of the ground-truth value."""
    value_candidate_indices = [quad.value_index for quad in quads]
    try:
        target_slot = value_candidate_indices.index(target_value_index)
    except ValueError as exc:
        raise ValueError(
            f"target_value_index={target_value_index} not among quad value indices "
            f"{value_candidate_indices}"
        ) from exc
    return value_candidate_indices, target_slot


def verify_slot_pointer_tokens(
    tokens: Sequence[int],
    *,
    question_index: int | None = None,
    expected_value_index: int | None = None,
) -> int:
    """Run structural checks and return traced value index."""
    if len(tokens) < 8:
        raise ValueError("sequence too short")
    q_idx = len(tokens) - 1 if question_index is None else question_index
    if tokens[q_idx] not in SEMANTIC_TOKENS:
        raise ValueError("question slot must hold a semantic addr token")

    for pos, tok in enumerate(tokens[:q_idx]):
        if tok == PAD_TOKEN_ID:
            raise ValueError(f"pad token in content at position {pos}")

    # Validate every quad delimiter alignment in content.
    i = 0
    while i < q_idx:
        if tokens[i] in SEMANTIC_TOKENS and i + 3 < q_idx and tokens[i + 1] == TOK_SEMICOLON:
            if tokens[i + 2] not in SEMANTIC_TOKENS or tokens[i + 3] != TOK_COMMA:
                raise ValueError(f"malformed quad starting at {i}")
            if tokens[i] == tokens[i + 2]:
                raise ValueError(f"addr and value identical within quad at {i}")
            i += 4
            continue
        if tokens[i] not in HAY_TOKENS:
            raise ValueError(f"unexpected token {tokens[i]} at content position {i}")
        i += 1

    traced = trace_value_index(tokens, q_idx)
    if expected_value_index is not None and traced != expected_value_index:
        raise ValueError(f"trace index {traced} != expected {expected_value_index}")
    return traced
