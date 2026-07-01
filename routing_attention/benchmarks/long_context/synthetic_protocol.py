"""
Hop-first synthetic NIAH protocol (v2).

1. Build the hop / needle graph first (fixed hop count from config).
2. Add verified-safe distractors (no accidental traceable alternate paths).
3. Generate collision-free hay, then scatter needles at random positions.
4. Query suffix is always appended at sequence end (``at_end``).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from routing_attention.benchmarks.long_context.tasks import TaskPayload

# Record / query tokens (fixed vocabulary).
ACTIVE_TAG = "ACTIVE"
ADDR_TAG = "ADDR"
VAL_TAG = "VAL"
PTR_TAG = "PTR"
QUERY_TAG = "QUERY"
Q_TAG = "Q"
PASSKEY_TAG = "PASSKEY"

POINTER_KEYS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ACTIVE_TAG_KEY_COLLISIONS = frozenset(
    k for k in POINTER_KEYS if f"{k} " in f"{ACTIVE_TAG} "
)


def _digit_value(rng: random.Random, width: int = 1) -> str:
    if width == 1:
        return str(rng.randint(0, 9))
    lo = 10 ** (width - 1)
    hi = (10**width) - 1
    return str(rng.randint(lo, hi))


def _distinct_digit_values(
    rng: random.Random,
    *,
    exclude: str,
    count: int,
    width: int = 1,
    max_width: int = 6,
) -> list[str]:
    if count <= 0:
        return []
    w = max(1, width)
    while w <= max_width:
        if w == 1:
            pool = [str(d) for d in range(10) if str(d) != exclude]
        else:
            lo = 10 ** (w - 1)
            hi = (10**w) - 1
            pool = [str(d) for d in range(lo, hi + 1) if str(d) != exclude]
        if len(pool) >= count:
            return rng.sample(pool, count)
        w += 1
    raise ValueError(f"cannot sample {count} distinct digit values excluding {exclude!r}")


def _pick_key(rng: random.Random, exclude: set[str] | None = None) -> str:
    pool = [k for k in POINTER_KEYS if not exclude or k not in exclude]
    return rng.choice(pool)


def _pick_addr(rng: random.Random, exclude: set[int] | None = None) -> int:
    for _ in range(64):
        addr = rng.randint(10, 999)
        if exclude is None or addr not in exclude:
            return addr
    raise RuntimeError("failed to sample unique address")


def _segment(text: str) -> str:
    return text if text.endswith(".") else f"{text}."

PROTOCOL_VERSION = 2

_PTR_RE = re.compile(rf"{ADDR_TAG} (\d+) {PTR_TAG} (\d+)")
_VAL_RE = re.compile(rf"{ADDR_TAG} (\d+) {VAL_TAG} (\S+?)\.")
@dataclass(frozen=True)
class PtrGraph:
    ptr_edges: dict[int, int]
    terminal_vals: dict[int, str]


def parse_ptr_graph(text: str) -> PtrGraph:
    """Parse ADDR/PTR/VAL records from haystack text."""
    ptr_edges: dict[int, int] = {}
    terminal_vals: dict[int, str] = {}
    for m in _PTR_RE.finditer(text):
        src, dst = int(m.group(1)), int(m.group(2))
        if src in ptr_edges and ptr_edges[src] != dst:
            ptr_edges[src] = -1  # mark ambiguous
        else:
            ptr_edges[src] = dst
    for m in _VAL_RE.finditer(text):
        addr, val = int(m.group(1)), m.group(2)
        if addr in terminal_vals and terminal_vals[addr] != val:
            terminal_vals[addr] = ""
        else:
            terminal_vals[addr] = val
    return PtrGraph(ptr_edges=ptr_edges, terminal_vals=terminal_vals)


def trace_ptr_chain(text: str, start_addr: int, *, max_hops: int = 16) -> str | None:
    """Deterministic pointer hop tracer; returns terminal VAL or None."""
    graph = parse_ptr_graph(text)
    addr = start_addr
    visited: set[int] = set()
    for _ in range(max_hops):
        if addr in visited:
            return None
        visited.add(addr)
        if addr in graph.ptr_edges:
            nxt = graph.ptr_edges[addr]
            if nxt < 0:
                return None
            addr = nxt
            continue
        if addr in graph.terminal_vals:
            val = graph.terminal_vals[addr]
            return val if val else None
        return None
    return None


def reachable_terminal_values(text: str, start_addr: int, *, max_hops: int = 16) -> set[str]:
    """All terminal VAL strings reachable from ``start_addr`` via PTR edges (BFS)."""
    graph = parse_ptr_graph(text)
    seen: set[int] = set()
    queue = [start_addr]
    terminals: set[str] = set()
    while queue:
        addr = queue.pop(0)
        if addr in seen:
            continue
        seen.add(addr)
        if addr in graph.ptr_edges:
            nxt = graph.ptr_edges[addr]
            if nxt >= 0:
                queue.append(nxt)
        if addr in graph.terminal_vals and graph.terminal_vals[addr]:
            terminals.add(graph.terminal_vals[addr])
    return terminals


def verify_ptr_chain_traceable(text: str, start_addr: int, expected_value: str) -> bool:
    if trace_ptr_chain(text, start_addr) != expected_value:
        return False
    terminals = reachable_terminal_values(text, start_addr)
    return terminals == {expected_value}


def trace_pointer_active(text: str, key: str) -> str | None:
    pat = re.compile(rf"{re.escape(ACTIVE_TAG)} {re.escape(key)} (\S+?)\.")
    hits = pat.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_pointer_unique(text: str, key: str) -> str | None:
    pat = re.compile(rf"(?<!{re.escape(ACTIVE_TAG)} ){re.escape(key)} (\S+?)\.")
    hits = pat.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_pointer_first(text: str, key: str) -> str | None:
    """Return the value from the first ``KEY val.`` row in reading order (not ACTIVE-tagged)."""
    pat = re.compile(rf"(?<!{re.escape(ACTIVE_TAG)} ){re.escape(key)} (\S+?)\.")
    hits = pat.findall(text)
    if not hits:
        return None
    return hits[0]


def trace_addr_val(text: str, addr: int) -> str | None:
    pat = re.compile(rf"{re.escape(ADDR_TAG)} {addr} {re.escape(VAL_TAG)} (\S+?)\.")
    hits = pat.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_addr_val_last(text: str, addr: int) -> str | None:
    """Return the value from the last ``ADDR addr VAL v.`` row in reading order."""
    pat = re.compile(rf"{re.escape(ADDR_TAG)} {addr} {re.escape(VAL_TAG)} (\S+?)\.")
    hits = pat.findall(text)
    if not hits:
        return None
    return hits[-1]


def trace_addr_val_first(text: str, addr: int) -> str | None:
    """Return the value from the first ``ADDR addr VAL v.`` row in reading order."""
    pat = re.compile(rf"{re.escape(ADDR_TAG)} {addr} {re.escape(VAL_TAG)} (\S+?)\.")
    hits = pat.findall(text)
    if not hits:
        return None
    return hits[0]


def trace_addr_val_middle(text: str, addr: int) -> str | None:
    """Return the value from the middle ``ADDR addr VAL v.`` row in reading order."""
    pat = re.compile(rf"{re.escape(ADDR_TAG)} {addr} {re.escape(VAL_TAG)} (\S+?)\.")
    hits = pat.findall(text)
    if not hits:
        return None
    return hits[len(hits) // 2]


_PASSKEY_RE = re.compile(rf"{re.escape(PASSKEY_TAG)} (\S+?)\.")


def trace_passkey_copy(text: str) -> str | None:
    """Single ``PASSKEY value.`` row — return its value."""
    hits = _PASSKEY_RE.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_passkey_distractor(text: str, target: str) -> str | None:
    """Target passkey value must appear in exactly one ``PASSKEY`` row."""
    hits = _PASSKEY_RE.findall(text)
    if hits.count(target) != 1:
        return None
    return target


def trace_addr_val_active(text: str, addr: int) -> str | None:
    pat = re.compile(
        rf"{re.escape(ACTIVE_TAG)} {re.escape(ADDR_TAG)} {addr} {re.escape(VAL_TAG)} (\S+?)\."
    )
    hits = pat.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_task_answer(text: str, payload: TaskPayload) -> str | None:
    task = payload.task_type
    meta = payload.metadata
    if task == "ptr_chain":
        return trace_ptr_chain(text, int(meta["chain"][0]))
    if task == "pointer_active":
        return trace_pointer_active(text, str(meta["key"]))
    if task == "pointer_unique":
        return trace_pointer_unique(text, str(meta["key"]))
    if task == "pointer_conflict_first":
        return trace_pointer_first(text, str(meta["key"]))
    if task == "addr_val":
        return trace_addr_val(text, int(meta["addr"]))
    if task == "addr_val_conflict":
        return trace_addr_val_last(text, int(meta["addr"]))
    if task == "addr_val_conflict_first":
        return trace_addr_val_first(text, int(meta["addr"]))
    if task == "addr_val_conflict_middle":
        return trace_addr_val_middle(text, int(meta["addr"]))
    if task == "passkey_copy":
        return trace_passkey_copy(text)
    if task == "passkey_distractor":
        return trace_passkey_distractor(text, str(meta["target"]))
    if task == "pointer_unique_copy":
        return trace_pointer_unique(text, str(meta["key"]))
    if task == "addr_val_active":
        return trace_addr_val_active(text, int(meta["addr"]))
    if task in ("massive_addr_val", "mqar_addr_val"):
        query_addrs = meta.get("query_addrs")
        if meta.get("mqar_supervise_all_queries") and query_addrs:
            vals: list[str] = []
            for addr in query_addrs:
                v = trace_addr_val(text, int(addr))
                if v is None:
                    return None
                vals.append(v)
            return " ".join(vals)
        if query_addrs:
            return trace_addr_val(text, int(query_addrs[-1]))
        return trace_addr_val(text, int(meta["addr"]))
    return None


def verify_task_traceable(text: str, payload: TaskPayload) -> bool:
    traced = trace_task_answer(text, payload)
    if traced != payload.expected_answer:
        return False
    if payload.task_type == "ptr_chain":
        return verify_ptr_chain_traceable(
            text, int(payload.metadata["chain"][0]), payload.expected_answer
        )
    return True


def collision_tokens(payload: TaskPayload) -> list[str]:
    """Substrings that must not appear in hay filler."""
    tokens: list[str] = list(payload.collision_check_strings())
    for seg in payload.needle_segments:
        body = seg.rstrip(".")
        for part in body.split():
            if part not in tokens:
                tokens.append(part)
    for tag in (ACTIVE_TAG, ADDR_TAG, VAL_TAG, PTR_TAG, QUERY_TAG, Q_TAG, PASSKEY_TAG):
        if tag not in tokens:
            tokens.append(tag)
    for forbidden in payload.metadata.get("filler_forbidden", []):
        if forbidden and forbidden not in tokens:
            tokens.append(str(forbidden))
    return [t for t in tokens if t]


def hay_chunk_is_clean(chunk: str, payload: TaskPayload) -> bool:
    if not chunk:
        return True
    for token in collision_tokens(payload):
        if token and token in chunk:
            return False
    if re.search(r"[A-Za-z0-9]", chunk):
        return False
    return True


def _even_spread_positions(
    seg_lens: list[int],
    *,
    pos_min: int,
    pos_max: int,
) -> list[int] | None:
    """Deterministic non-overlapping layout with evenly distributed slack."""
    if not seg_lens:
        return []
    total_needle = sum(seg_lens)
    span = pos_max - pos_min
    if total_needle > span:
        return None
    n = len(seg_lens)
    slack = span - total_needle
    gaps = [slack // (n + 1)] * (n + 1)
    for i in range(slack % (n + 1)):
        gaps[i] += 1
    positions: list[int] = []
    cursor = pos_min + gaps[0]
    for i, length in enumerate(seg_lens):
        if cursor + length > pos_max:
            return None
        positions.append(cursor)
        cursor += length + gaps[i + 1]
    return positions


def assign_non_overlapping_positions(
    rng: random.Random,
    *,
    seg_lens: list[int],
    pos_min: int,
    pos_max: int,
    max_attempts: int = 256,
) -> list[int] | None:
    """Pick start indices in ``[pos_min, pos_max)`` without overlap."""
    if not seg_lens:
        return []
    total = sum(seg_lens) + max(0, len(seg_lens) - 1)
    if pos_max - pos_min < total:
        return None
    for _ in range(max_attempts):
        order = list(range(len(seg_lens)))
        rng.shuffle(order)
        placed: list[tuple[int, int]] = []
        ok = True
        for idx in order:
            length = seg_lens[idx]
            hi = pos_max - length
            if hi < pos_min:
                ok = False
                break
            start = rng.randint(pos_min, hi)
            end = start + length
            for ps, pe in placed:
                if not (end <= ps or start >= pe):
                    ok = False
                    break
            if not ok:
                break
            placed.append((start, end))
        if ok and len(placed) == len(seg_lens):
            positions = [0] * len(seg_lens)
            for (idx, (start, _)) in zip(order, placed):
                positions[idx] = start
            return positions
    return _even_spread_positions(seg_lens, pos_min=pos_min, pos_max=pos_max)


def assemble_scattered_haystack(
    *,
    haystack_len: int,
    segments: list[str],
    positions: list[int],
    gap_filler: list[str],
) -> str:
    """Build exact-length haystack from pre-generated gap strings and needles."""
    if len(segments) != len(positions) or len(gap_filler) != len(segments) + 1:
        raise ValueError("segments/positions/gap_filler length mismatch")
    parts: list[str] = [gap_filler[0]]
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(" ")
        parts.append(gap_filler[i + 1])
    body = "".join(parts)
    if len(body) != haystack_len:
        raise RuntimeError(f"assembled haystack {len(body)} != {haystack_len}")
    return body


def needle_span_stats(positions: list[int], seg_lens: list[int]) -> dict[str, float]:
    if not positions:
        return {"span": 0.0, "std": 0.0, "min_gap": 0.0}
    centers = [p + l / 2.0 for p, l in zip(positions, seg_lens)]
    span = max(centers) - min(centers)
    mean = sum(centers) / len(centers)
    var = sum((c - mean) ** 2 for c in centers) / len(centers)
    sorted_starts = sorted(positions)
    gaps = [
        sorted_starts[i + 1] - (sorted_starts[i] + seg_lens[i])
        for i in range(len(sorted_starts) - 1)
    ]
    min_gap = float(min(gaps)) if gaps else 0.0
    return {"span": span, "std": var**0.5, "min_gap": min_gap}


def _protocol_metadata(
    *,
    hop_count: int,
    num_distractors: int,
    extra: dict | None = None,
) -> dict:
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "synthetic_protocol": "hop_first_scattered_at_end",
        "hop_count": hop_count,
        "num_distractors": num_distractors,
        "trace_verified": True,
    }
    if extra:
        meta.update(extra)
    return meta


def generate_ptr_chain_hop_first(
    rng: random.Random,
    *,
    hop_count: int,
    num_distractors: int,
) -> TaskPayload:
    hops = max(1, hop_count)
    addrs: list[int] = []
    used: set[int] = set()
    for _ in range(hops):
        addrs.append(_pick_addr(rng, exclude=used))
        used.add(addrs[-1])
    value = _digit_value(rng, width=2)
    gold_segments: list[str] = []
    for i in range(hops - 1):
        gold_segments.append(_segment(f"{ADDR_TAG} {addrs[i]} {PTR_TAG} {addrs[i + 1]}"))
    gold_segments.append(_segment(f"{ADDR_TAG} {addrs[-1]} {VAL_TAG} {value}"))

    distractors: list[str] = []
    for _ in range(max(0, num_distractors)):
        decoy_addr = _pick_addr(rng, exclude=used)
        used.add(decoy_addr)
        decoy_val = _digit_value(rng, width=2)
        while decoy_val == value:
            decoy_val = _digit_value(rng, width=2)
        distractors.append(_segment(f"{ADDR_TAG} {decoy_addr} {VAL_TAG} {decoy_val}"))

    segments = gold_segments + distractors
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addrs[0]}"
    probe = " ".join(segments)
    if not verify_ptr_chain_traceable(probe, addrs[0], value):
        raise ValueError("ptr_chain hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="ptr_chain",
        metadata=_protocol_metadata(
            hop_count=hops,
            num_distractors=len(distractors),
            extra={
                "level": 4,
                "ptr_hops": hops - 1,
                "chain": addrs,
                "value": value,
                "collision_checks": [value, f"{ADDR_TAG} {addrs[-1]} {VAL_TAG} {value}"],
                "gold_segments": gold_segments,
            },
        ),
    )


def generate_pointer_active_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
) -> TaskPayload:
    key = _pick_key(rng, exclude=set(_ACTIVE_TAG_KEY_COLLISIONS))
    value = _digit_value(rng)
    n = max(0, num_distractors)
    fake_width = 2 if n > 9 else 1
    fakes = _distinct_digit_values(rng, exclude=value, count=n, width=fake_width) if n else []
    distractor_segs = [_segment(f"{key} {fv}") for fv in sorted(fakes)]
    active_seg = _segment(f"{ACTIVE_TAG} {key} {value}")
    segments = distractor_segs + [active_seg]
    rng.shuffle(segments)
    question = f"{Q_TAG} {key}"
    gold_needle = f"{ACTIVE_TAG} {key} {value}"
    probe = " ".join(segments)
    if trace_pointer_active(probe, key) != value:
        raise ValueError("pointer_active hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="pointer_active",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(fakes),
            extra={
                "level": 1,
                "key": key,
                "value": value,
                "gold_needle": gold_needle,
                "collision_checks": [value, gold_needle],
                "filler_forbidden": [f"{key} ", f"{ACTIVE_TAG} "],
            },
        ),
    )


def generate_pointer_unique_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
    answer_digit_width: int = 1,
) -> TaskPayload:
    width = max(1, int(answer_digit_width))
    key = _pick_key(rng)
    value = _digit_value(rng, width=width)
    needle = _segment(f"{key} {value}")
    decoys: list[str] = []
    used_keys: set[str] = {key}
    for _ in range(max(0, num_distractors)):
        k = _pick_key(rng, exclude=used_keys)
        used_keys.add(k)
        decoys.append(_segment(f"{k} {_digit_value(rng, width=width)}"))
    segments = decoys + [needle]
    rng.shuffle(segments)
    question = f"{Q_TAG} {key}"
    probe = " ".join(segments)
    if trace_pointer_unique(probe, key) != value:
        raise ValueError("pointer_unique hop-first spec failed trace verification")

    task_type = "pointer_unique"
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type=task_type,
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(decoys),
            extra={
                "level": 0,
                "key": key,
                "value": value,
                "answer_digit_width": width,
                "collision_checks": [value, f"{key} {value}"],
                "unique_key": key,
                "filler_forbidden": [f"{key} "],
            },
        ),
    )


def generate_pointer_conflict_first_hop_first(
    rng: random.Random,
    *,
    num_conflict_rows: int,
    num_distractors: int = 1,
    answer_digit_width: int = 1,
) -> TaskPayload:
    """
    Same query key repeated with conflicting values — first row in text order wins.

    Optional decoy-key rows (different keys) enable multi-segment scatter.
    """
    width = max(1, int(answer_digit_width))
    n_rows = max(2, int(num_conflict_rows))
    key = _pick_key(rng)
    values = _distinct_digit_values(rng, exclude="", count=n_rows, width=width)
    conflict_segs = [_segment(f"{key} {v}") for v in values]
    decoys: list[str] = []
    used_keys: set[str] = {key}
    for _ in range(max(0, int(num_distractors))):
        k = _pick_key(rng, exclude=used_keys)
        used_keys.add(k)
        decoys.append(_segment(f"{k} {_digit_value(rng, width=width)}"))
    segments = decoys + conflict_segs
    rng.shuffle(segments)
    question = f"{Q_TAG} {key}"
    probe = " ".join(segments)
    expected = trace_pointer_first(probe, key)
    if expected is None:
        raise ValueError("pointer_conflict_first hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=expected,
        task_type="pointer_conflict_first",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(decoys),
            extra={
                "level": 0,
                "key": key,
                "value": expected,
                "answer_digit_width": width,
                "num_conflict_rows": n_rows,
                "conflict_policy": "first",
                "collision_checks": [expected, f"{key} {expected}"],
                "filler_forbidden": [f"{key} "],
            },
        ),
    )


def generate_pointer_unique_copy_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
    answer_digit_width: int = 4,
) -> TaskPayload:
    """Multi-digit ``KEY VAL`` selective copy (L0 variant with wider answers)."""
    width = max(2, min(int(answer_digit_width), 6))
    payload = generate_pointer_unique_hop_first(
        rng,
        num_distractors=num_distractors,
        answer_digit_width=width,
    )
    return TaskPayload(
        needle_segments=payload.needle_segments,
        question=payload.question,
        expected_answer=payload.expected_answer,
        task_type="pointer_unique_copy",
        metadata={
            **payload.metadata,
            "task_type": "pointer_unique_copy",
        },
    )


def generate_addr_val_conflict_hop_first(
    rng: random.Random,
    *,
    num_conflict_rows: int,
    answer_digit_width: int = 1,
) -> TaskPayload:
    """Same address repeated with different values; last row in text order wins."""
    width = max(1, int(answer_digit_width))
    n = max(2, int(num_conflict_rows))
    addr = _pick_addr(rng)
    values = _distinct_digit_values(rng, exclude="", count=n, width=width)
    segments = [_segment(f"{ADDR_TAG} {addr} {VAL_TAG} {v}") for v in values]
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addr}"
    probe = " ".join(segments)
    value = trace_addr_val_last(probe, addr)
    if value is None:
        raise ValueError("addr_val_conflict hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="addr_val_conflict",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=0,
            extra={
                "level": 2,
                "addr": addr,
                "value": value,
                "num_conflict_rows": n,
                "collision_checks": [value, f"{ADDR_TAG} {addr} {VAL_TAG} {value}"],
            },
        ),
    )


def _generate_addr_val_conflict_variant_hop_first(
    rng: random.Random,
    *,
    num_conflict_rows: int,
    answer_digit_width: int,
    task_type: str,
    trace_fn,
    conflict_policy: str,
) -> TaskPayload:
    width = max(1, int(answer_digit_width))
    n = max(2, int(num_conflict_rows))
    addr = _pick_addr(rng)
    values = _distinct_digit_values(rng, exclude="", count=n, width=width)
    segments = [_segment(f"{ADDR_TAG} {addr} {VAL_TAG} {v}") for v in values]
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addr}"
    probe = " ".join(segments)
    value = trace_fn(probe, addr)
    if value is None:
        raise ValueError(f"{task_type} hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type=task_type,
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=0,
            extra={
                "level": 2,
                "addr": addr,
                "value": value,
                "num_conflict_rows": n,
                "conflict_policy": conflict_policy,
                "collision_checks": [value, f"{ADDR_TAG} {addr} {VAL_TAG} {value}"],
            },
        ),
    )


def generate_addr_val_conflict_first_hop_first(
    rng: random.Random,
    *,
    num_conflict_rows: int,
    answer_digit_width: int = 1,
) -> TaskPayload:
    """Same address repeated; first row in text order wins."""
    return _generate_addr_val_conflict_variant_hop_first(
        rng,
        num_conflict_rows=num_conflict_rows,
        answer_digit_width=answer_digit_width,
        task_type="addr_val_conflict_first",
        trace_fn=trace_addr_val_first,
        conflict_policy="first",
    )


def generate_addr_val_conflict_middle_hop_first(
    rng: random.Random,
    *,
    num_conflict_rows: int,
    answer_digit_width: int = 1,
) -> TaskPayload:
    """Same address repeated; middle row in text order wins."""
    return _generate_addr_val_conflict_variant_hop_first(
        rng,
        num_conflict_rows=num_conflict_rows,
        answer_digit_width=answer_digit_width,
        task_type="addr_val_conflict_middle",
        trace_fn=trace_addr_val_middle,
        conflict_policy="middle",
    )


def generate_passkey_copy_hop_first(
    rng: random.Random,
    *,
    answer_digit_width: int = 5,
) -> TaskPayload:
    """Single scattered passkey needle — multi-digit exact copy."""
    width = max(4, min(int(answer_digit_width), 6))
    value = _digit_value(rng, width=width)
    needle = _segment(f"{PASSKEY_TAG} {value}")
    question = f"{QUERY_TAG} {PASSKEY_TAG}"
    probe = needle
    if trace_passkey_copy(probe) != value:
        raise ValueError("passkey_copy hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=[needle],
        question=question,
        expected_answer=value,
        task_type="passkey_copy",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=0,
            extra={
                "level": 0,
                "target": value,
                "answer_digit_width": width,
                "collision_checks": [value, f"{PASSKEY_TAG} {value}"],
                "filler_forbidden": [f"{PASSKEY_TAG} "],
            },
        ),
    )


def generate_passkey_distractor_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
    answer_digit_width: int = 5,
) -> TaskPayload:
    """Multiple passkey needles; copy the unique target value."""
    width = max(4, min(int(answer_digit_width), 6))
    n = max(1, int(num_distractors))
    target = _digit_value(rng, width=width)
    decoys = _distinct_digit_values(rng, exclude=target, count=n, width=width)
    segments = [_segment(f"{PASSKEY_TAG} {d}") for d in decoys]
    insert_at = rng.randint(0, len(segments))
    segments.insert(insert_at, _segment(f"{PASSKEY_TAG} {target}"))
    question = f"{QUERY_TAG} {PASSKEY_TAG}"
    probe = " ".join(segments)
    if trace_passkey_distractor(probe, target) != target:
        raise ValueError("passkey_distractor hop-first spec failed trace verification")

    all_values = [target] + decoys
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=target,
        task_type="passkey_distractor",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(decoys),
            extra={
                "level": 1,
                "target": target,
                "answer_digit_width": width,
                "collision_checks": list(dict.fromkeys(all_values))
                + [f"{PASSKEY_TAG} {v}" for v in all_values],
                "filler_forbidden": [f"{PASSKEY_TAG} "],
            },
        ),
    )


def generate_addr_val_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
    answer_digit_width: int = 2,
) -> TaskPayload:
    width = max(1, int(answer_digit_width))
    addr = _pick_addr(rng)
    value = _digit_value(rng, width=width)
    needle = _segment(f"{ADDR_TAG} {addr} {VAL_TAG} {value}")
    used = {addr}
    decoys: list[str] = []
    for _ in range(max(0, num_distractors)):
        a = _pick_addr(rng, exclude=used)
        used.add(a)
        decoys.append(_segment(f"{ADDR_TAG} {a} {VAL_TAG} {_digit_value(rng, width=width)}"))
    segments = decoys + [needle]
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addr}"
    probe = " ".join(segments)
    if trace_addr_val(probe, addr) != value:
        raise ValueError("addr_val hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="addr_val",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(decoys),
            extra={
                "level": 2,
                "addr": addr,
                "value": value,
                "collision_checks": [value, f"{ADDR_TAG} {addr} {VAL_TAG} {value}"],
            },
        ),
    )


def generate_addr_val_active_hop_first(
    rng: random.Random,
    *,
    num_distractors: int,
) -> TaskPayload:
    addr = _pick_addr(rng)
    value = _digit_value(rng, width=2)
    n = max(0, num_distractors)
    fakes = _distinct_digit_values(rng, exclude=value, count=n, width=2) if n else []
    distractor_segs = [_segment(f"{ADDR_TAG} {addr} {VAL_TAG} {fv}") for fv in sorted(fakes)]
    active_seg = _segment(f"{ACTIVE_TAG} {ADDR_TAG} {addr} {VAL_TAG} {value}")
    segments = distractor_segs + [active_seg]
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addr}"
    gold_needle = f"{ACTIVE_TAG} {ADDR_TAG} {addr} {VAL_TAG} {value}"
    probe = " ".join(segments)
    if trace_addr_val_active(probe, addr) != value:
        raise ValueError("addr_val_active hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="addr_val_active",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=len(fakes),
            extra={
                "level": 3,
                "addr": addr,
                "value": value,
                "gold_needle": gold_needle,
                "collision_checks": [value, gold_needle],
            },
        ),
    )


def _sample_unique_kv_bindings(
    rng: random.Random,
    *,
    num_kv_pairs: int,
    answer_digit_width: int,
) -> tuple[list[int], list[str], list[str]]:
    """Sample ``num_kv_pairs`` unique ADDR→VAL rows (all real bindings, no decoys)."""
    n = max(2, min(int(num_kv_pairs), 990))
    width = max(1, int(answer_digit_width))
    addrs: list[int] = []
    used_addrs: set[int] = set()
    for _ in range(n):
        addr = _pick_addr(rng, exclude=used_addrs)
        used_addrs.add(addr)
        addrs.append(addr)
    values = [_digit_value(rng, width=width) for _ in range(n)]
    segments = [_segment(f"{ADDR_TAG} {a} {VAL_TAG} {v}") for a, v in zip(addrs, values)]
    return addrs, values, segments


def generate_massive_addr_val_hop_first(
    rng: random.Random,
    *,
    num_kv_pairs: int,
    answer_digit_width: int = 2,
) -> TaskPayload:
    """Many independent ``ADDR n VAL v`` rows; query one address among N bindings."""
    addrs, values, segments = _sample_unique_kv_bindings(
        rng,
        num_kv_pairs=num_kv_pairs,
        answer_digit_width=answer_digit_width,
    )
    n = len(addrs)
    rng.shuffle(segments)
    target_idx = rng.randint(0, n - 1)
    target_addr = addrs[target_idx]
    target_value = values[target_idx]
    question = f"{QUERY_TAG} {target_addr}"
    probe = " ".join(segments)
    if trace_addr_val(probe, target_addr) != target_value:
        raise ValueError("massive_addr_val hop-first spec failed trace verification")

    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=target_value,
        task_type="massive_addr_val",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=0,
            extra={
                "level": 5,
                "addr": target_addr,
                "value": target_value,
                "num_kv_pairs": n,
                "answer_digit_width": max(1, int(answer_digit_width)),
                "collision_checks": list(dict.fromkeys(values))
                + [f"{ADDR_TAG} {a} {VAL_TAG} {v}" for a, v in zip(addrs, values)],
            },
        ),
    )


def generate_mqar_addr_val_hop_first(
    rng: random.Random,
    *,
    num_kv_pairs: int,
    answer_digit_width: int = 1,
    num_queries: int = 1,
    supervise_all_queries: bool = False,
) -> TaskPayload:
    """
    MQAR-style many-binding lookup: N real ``ADDR n VAL v`` rows, query suffix at sequence end.

    When ``supervise_all_queries`` is false, supervision/eval target is the **last** query's value.
    When true, ``expected_answer`` is space-separated values for every query in suffix order.
    """
    addrs, values, segments = _sample_unique_kv_bindings(
        rng,
        num_kv_pairs=num_kv_pairs,
        answer_digit_width=answer_digit_width,
    )
    n = len(addrs)
    q = max(1, min(int(num_queries), n))
    rng.shuffle(segments)
    query_indices = rng.sample(range(n), q)
    query_addrs = [addrs[i] for i in query_indices]
    supervise_all = bool(supervise_all_queries)
    if supervise_all:
        answer_values = [values[i] for i in query_indices]
        target_addr = query_addrs[-1]
        target_value = values[query_indices[-1]]
        expected_answer = " ".join(answer_values)
    else:
        target_idx = query_indices[-1]
        target_addr = addrs[target_idx]
        target_value = values[target_idx]
        expected_answer = target_value
    question = " ".join(f"{QUERY_TAG} {a}" for a in query_addrs)
    probe = " ".join(segments)
    if supervise_all:
        traced = trace_task_answer(
            probe,
            TaskPayload(
                needle_segments=segments,
                question=question,
                expected_answer=expected_answer,
                task_type="mqar_addr_val",
                metadata={
                    "query_addrs": query_addrs,
                    "addr": target_addr,
                    "mqar_supervise_all_queries": True,
                },
            ),
        )
        if traced != expected_answer:
            raise ValueError("mqar_addr_val hop-first spec failed trace verification")
    elif trace_addr_val(probe, target_addr) != target_value:
        raise ValueError("mqar_addr_val hop-first spec failed trace verification")

    bindings = [{"addr": a, "value": v} for a, v in zip(addrs, values)]
    width = max(1, int(answer_digit_width))
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=expected_answer,
        task_type="mqar_addr_val",
        metadata=_protocol_metadata(
            hop_count=1,
            num_distractors=0,
            extra={
                "level": 6,
                "protocol": "mqar_addr_val_v1",
                "addr": target_addr,
                "value": target_value,
                "num_kv_pairs": n,
                "num_queries": q,
                "query_addrs": query_addrs,
                "bindings": bindings,
                "answer_digit_width": width,
                "mqar_supervise_all_queries": supervise_all,
                "collision_checks": list(dict.fromkeys(values))
                + [f"{ADDR_TAG} {a} {VAL_TAG} {v}" for a, v in zip(addrs, values)],
            },
        ),
    )
