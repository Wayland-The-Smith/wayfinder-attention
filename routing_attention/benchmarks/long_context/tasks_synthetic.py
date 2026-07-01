"""
Synthetic pointer / address NIAH tasks (L0–L4).

Minimal vocabulary, no natural language. Each level tests a distinct retrieval skill.

L0 pointer_unique   — single ``KEY VAL`` row; query key appears once in context.
L1 pointer_active   — repeated keys; only ``ACTIVE KEY VAL`` is authoritative.
L2 addr_val         — ``ADDR n VAL v`` keyed lookup.
L3 addr_val_active  — repeated address rows; only ``ACTIVE ADDR n VAL v`` counts.
L4 ptr_chain        — multi-hop ``ADDR a PTR b`` … ``ADDR z VAL v`` chain.
L5 massive_addr_val — N independent ``ADDR n VAL v`` bindings; tests KV capacity.
L6 mqar_addr_val      — N real bindings, query-only suffix, 1-digit values (MQAR calibration).
L7 addr_val_conflict* — same address, conflicting values (last / first / middle wins).
L8 passkey_copy       — single ``PASSKEY value.`` multi-digit exact copy.
L9 passkey_distractor — multiple passkeys; copy the unique target value.
L10 pointer_unique_copy — multi-digit ``KEY VAL`` selective copy.
"""

from __future__ import annotations

import random
import string

from routing_attention.benchmarks.long_context.tasks import (
    TaskPayload,
    enforce_max_answer_chars,
    question_leaks_answer,
)

# Record / query tokens (fixed vocabulary).
ACTIVE_TAG = "ACTIVE"
ADDR_TAG = "ADDR"
VAL_TAG = "VAL"
PTR_TAG = "PTR"
QUERY_TAG = "QUERY"
Q_TAG = "Q"

POINTER_KEYS = tuple(string.ascii_uppercase)  # A–Z

# Keys whose ``{K} `` substring appears inside a reserved tag (e.g. ``E`` in ``ACTIVE``).
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
    """Sample ``count`` unique digit strings != ``exclude``, widening width if needed."""
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
    raise ValueError(
        f"cannot sample {count} distinct distractor values excluding {exclude!r} "
        f"(max_width={max_width})"
    )


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


def _decoy_pointer_rows(rng: random.Random, *, exclude_key: str, count: int) -> list[str]:
    """Other-key ``KEY VAL`` rows (query key excluded — appears only in the true needle)."""
    rows: list[str] = []
    used_keys: set[str] = {exclude_key}
    for _ in range(count):
        k = _pick_key(rng, exclude=used_keys)
        used_keys.add(k)
        rows.append(_segment(f"{k} {_digit_value(rng)}"))
    return rows


def _decoy_addr_rows(
    rng: random.Random,
    *,
    exclude_addr: int | set[int] | None = None,
    exclude_values: set[str] | None = None,
    count: int,
) -> list[str]:
    """``ADDR n VAL v`` rows for addresses other than excluded target(s)."""
    excluded = {exclude_addr} if isinstance(exclude_addr, int) else set(exclude_addr or ())
    rows: list[str] = []
    used: set[int] = set(excluded)
    for _ in range(count):
        a = _pick_addr(rng, exclude=used)
        used.add(a)
        val = _digit_value(rng, width=2)
        if exclude_values:
            for _ in range(64):
                if val not in exclude_values:
                    break
                val = _digit_value(rng, width=2)
        rows.append(_segment(f"{ADDR_TAG} {a} {VAL_TAG} {val}"))
    return rows


def generate_pointer_unique(rng: random.Random, num_decoy_keys: int = 5) -> TaskPayload:
    """L0: unique query key — that key appears once; other keys may appear as decoys."""
    key = _pick_key(rng)
    value = _digit_value(rng)
    needle = _segment(f"{key} {value}")
    decoys = _decoy_pointer_rows(rng, exclude_key=key, count=max(0, num_decoy_keys))
    segments = decoys + [needle]
    rng.shuffle(segments)
    question = f"{Q_TAG} {key}"
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="pointer_unique",
        metadata={
            "level": 0,
            "key": key,
            "value": value,
            "collision_checks": [value, f"{key} {value}"],
            "unique_key": key,
            "filler_forbidden": [f"{key} "],
            "num_decoy_keys": len(decoys),
        },
    )


def generate_pointer_active(rng: random.Random, num_distractors: int = 6) -> TaskPayload:
    """L1: repeated key rows; only ``ACTIVE KEY VAL`` is correct."""
    key = _pick_key(rng, exclude=set(_ACTIVE_TAG_KEY_COLLISIONS))
    value = _digit_value(rng)
    n = max(2, num_distractors)
    fake_width = 2 if n > 9 else 1
    fakes = _distinct_digit_values(rng, exclude=value, count=n, width=fake_width)
    distractor_segs = [_segment(f"{key} {fv}") for fv in sorted(fakes)]
    active_seg = _segment(f"{ACTIVE_TAG} {key} {value}")
    insert_at = rng.randint(0, len(distractor_segs))
    segments = list(distractor_segs)
    segments.insert(insert_at, active_seg)
    question = f"{Q_TAG} {key}"
    gold_needle = f"{ACTIVE_TAG} {key} {value}"
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="pointer_active",
        metadata={
            "level": 1,
            "key": key,
            "value": value,
            "gold_needle": gold_needle,
            "collision_checks": [value, gold_needle],
            "num_distractors": len(fakes),
        },
    )


def generate_addr_val(rng: random.Random, num_decoy_addrs: int = 4) -> TaskPayload:
    """L2: ``ADDR addr VAL value`` — query names one address among optional decoy ADDR rows."""
    addr = _pick_addr(rng)
    value = _digit_value(rng, width=2)
    needle = _segment(f"{ADDR_TAG} {addr} {VAL_TAG} {value}")
    decoys = _decoy_addr_rows(rng, exclude_addr=addr, count=max(0, num_decoy_addrs))
    segments = decoys + [needle]
    rng.shuffle(segments)
    question = f"{QUERY_TAG} {addr}"
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="addr_val",
        metadata={
            "level": 2,
            "addr": addr,
            "value": value,
            "collision_checks": [value, f"{ADDR_TAG} {addr} {VAL_TAG} {value}"],
            "num_decoy_addrs": len(decoys),
        },
    )


def generate_addr_val_active(rng: random.Random, num_distractors: int = 6) -> TaskPayload:
    """L3: same address repeated; only ``ACTIVE ADDR addr VAL value`` is correct."""
    addr = _pick_addr(rng)
    value = _digit_value(rng, width=2)
    n = max(2, num_distractors)
    fakes = _distinct_digit_values(rng, exclude=value, count=n, width=2)
    distractor_segs = [_segment(f"{ADDR_TAG} {addr} {VAL_TAG} {fv}") for fv in sorted(fakes)]
    active_seg = _segment(f"{ACTIVE_TAG} {ADDR_TAG} {addr} {VAL_TAG} {value}")
    insert_at = rng.randint(0, len(distractor_segs))
    segments = list(distractor_segs)
    segments.insert(insert_at, active_seg)
    question = f"{QUERY_TAG} {addr}"
    gold_needle = f"{ACTIVE_TAG} {ADDR_TAG} {addr} {VAL_TAG} {value}"
    return TaskPayload(
        needle_segments=segments,
        question=question,
        expected_answer=value,
        task_type="addr_val_active",
        metadata={
            "level": 3,
            "addr": addr,
            "value": value,
            "gold_needle": gold_needle,
            "collision_checks": [value, gold_needle],
            "num_distractors": len(fakes),
        },
    )


def generate_ptr_chain(
    rng: random.Random,
    hop_count: int | None = None,
    hop_count_range: tuple[int, int] = (2, 4),
    num_decoy_addrs: int = 4,
    num_fake_ptrs: int = 2,
) -> TaskPayload:
    """
    L4: pointer chain ending in a value.

    ``hop_count`` is the number of addresses in the chain (terminal included).
    Pointer dereferences (PTR edges) = ``hop_count - 1``.

    2-address / 1-PTR example (``hop_count=2``)::

        ADDR 91 PTR 52.
        ADDR 52 VAL 3.

        QUERY 91  ->  3
    """
    if hop_count is None:
        lo, hi = hop_count_range
        hops = rng.randint(max(2, lo), max(2, hi))
    else:
        hops = max(2, min(hop_count, 6))
    addrs: list[int] = []
    used: set[int] = set()
    for _ in range(hops):
        addrs.append(_pick_addr(rng, exclude=used))
        used.add(addrs[-1])
    value = _digit_value(rng, width=2)
    segments: list[str] = []
    for i in range(hops - 1):
        segments.append(_segment(f"{ADDR_TAG} {addrs[i]} {PTR_TAG} {addrs[i + 1]}"))
    segments.append(_segment(f"{ADDR_TAG} {addrs[-1]} {VAL_TAG} {value}"))

    chain_set = set(addrs)
    fake_segs: list[str] = []
    fake_targets: set[int] = set()
    for _ in range(max(0, num_fake_ptrs)):
        wrong = _pick_addr(rng, exclude=chain_set | fake_targets)
        fake_targets.add(wrong)
        wrong_val = _digit_value(rng, width=2)
        while wrong_val == value:
            wrong_val = _digit_value(rng, width=2)
        fake_segs.append(_segment(f"{ADDR_TAG} {addrs[0]} {PTR_TAG} {wrong}"))
        fake_segs.append(_segment(f"{ADDR_TAG} {wrong} {VAL_TAG} {wrong_val}"))

    decoys = _decoy_addr_rows(
        rng,
        exclude_addr=chain_set | fake_targets,
        exclude_values={value},
        count=max(0, num_decoy_addrs),
    )
    all_segments = fake_segs + decoys + segments
    rng.shuffle(all_segments)
    question = f"{QUERY_TAG} {addrs[0]}"
    return TaskPayload(
        needle_segments=all_segments,
        question=question,
        expected_answer=value,
        task_type="ptr_chain",
        metadata={
            "level": 4,
            "hop_count": hops,
            "ptr_hops": max(0, hops - 1),
            "chain": addrs,
            "value": value,
            "num_fake_ptrs": len(fake_segs) // 2,
            "num_decoy_addrs": len(decoys),
            "collision_checks": [value, f"{ADDR_TAG} {addrs[-1]} {VAL_TAG} {value}"],
        },
    )


SYNTHETIC_TASK_GENERATORS = {
    "pointer_unique": lambda rng: generate_pointer_unique(rng, num_decoy_keys=5),
    "pointer_active": lambda rng: generate_pointer_active(rng, num_distractors=6),
    "addr_val": lambda rng: generate_addr_val(rng, num_decoy_addrs=4),
    "addr_val_active": lambda rng: generate_addr_val_active(rng, num_distractors=6),
    "ptr_chain": lambda rng: generate_ptr_chain(rng),
}

# All five levels are separate tasks; train and holdout share the same task set
# (disjoint procedural samples via seed vs holdout_seed — not a different split).
SYNTHETIC_TASK_TYPES = (
    "pointer_unique",
    "pointer_unique_copy",
    "pointer_conflict_first",
    "pointer_active",
    "addr_val",
    "addr_val_active",
    "ptr_chain",
    "massive_addr_val",
    "mqar_addr_val",
    "addr_val_conflict",
    "addr_val_conflict_first",
    "addr_val_conflict_middle",
    "passkey_copy",
    "passkey_distractor",
    "slot_pointer",
)

SYNTHETIC_PRIMARY_GATE_TASK_TYPES = SYNTHETIC_TASK_TYPES
SYNTHETIC_SECONDARY_TASK_TYPES: tuple[str, ...] = ()
SYNTHETIC_ALL_TASK_TYPES = SYNTHETIC_TASK_TYPES


def generate_synthetic_task(
    task_type: str,
    rng: random.Random,
    config_kwargs: dict | None = None,
) -> TaskPayload:
    if task_type not in SYNTHETIC_TASK_TYPES:
        raise ValueError(
            f"Unknown synthetic task {task_type!r}. Available: {list(SYNTHETIC_TASK_TYPES)}"
        )
    cfg = config_kwargs or {}
    max_answer = int(cfg.get("max_answer_chars", 64))
    hop_lo = int(cfg.get("synthetic_hop_count_min", cfg.get("synthetic_hop_count", 2)))
    hop_hi = int(cfg.get("synthetic_hop_count_max", cfg.get("synthetic_hop_count", 4)))
    hop_fixed = int(cfg.get("synthetic_hop_count", hop_lo))
    hop_lo = hop_hi = hop_fixed
    num_distractors = int(cfg.get("num_distractors", 6))
    num_decoy_keys = int(cfg.get("synthetic_decoy_keys", num_distractors))
    num_decoy_addrs = int(cfg.get("synthetic_decoy_addrs", num_distractors))
    num_kv_pairs = int(cfg.get("num_kv_pairs", 50))
    num_queries = int(cfg.get("num_queries", 1))
    answer_digit_width = int(cfg.get("answer_digit_width", 2))
    num_conflict_rows = int(cfg.get("synthetic_conflict_rows", 3))
    mqar_supervise_all = bool(cfg.get("mqar_supervise_all_queries", False))

    from routing_attention.benchmarks.long_context.synthetic_protocol import (
        generate_addr_val_active_hop_first,
        generate_addr_val_conflict_first_hop_first,
        generate_addr_val_conflict_hop_first,
        generate_addr_val_conflict_middle_hop_first,
        generate_addr_val_hop_first,
        generate_massive_addr_val_hop_first,
        generate_mqar_addr_val_hop_first,
        generate_passkey_copy_hop_first,
        generate_passkey_distractor_hop_first,
        generate_pointer_active_hop_first,
        generate_pointer_conflict_first_hop_first,
        generate_pointer_unique_copy_hop_first,
        generate_pointer_unique_hop_first,
        generate_ptr_chain_hop_first,
    )

    for _ in range(32):
        if task_type == "pointer_unique":
            payload = generate_pointer_unique_hop_first(
                rng,
                num_distractors=num_decoy_keys,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "pointer_unique_copy":
            payload = generate_pointer_unique_copy_hop_first(
                rng,
                num_distractors=num_decoy_keys,
                answer_digit_width=max(2, answer_digit_width),
            )
        elif task_type == "pointer_conflict_first":
            payload = generate_pointer_conflict_first_hop_first(
                rng,
                num_conflict_rows=num_conflict_rows,
                num_distractors=num_decoy_keys,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "pointer_active":
            payload = generate_pointer_active_hop_first(rng, num_distractors=num_distractors)
        elif task_type == "addr_val":
            payload = generate_addr_val_hop_first(
                rng,
                num_distractors=num_decoy_addrs,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "addr_val_conflict":
            payload = generate_addr_val_conflict_hop_first(
                rng,
                num_conflict_rows=num_conflict_rows,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "addr_val_conflict_first":
            payload = generate_addr_val_conflict_first_hop_first(
                rng,
                num_conflict_rows=num_conflict_rows,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "addr_val_conflict_middle":
            payload = generate_addr_val_conflict_middle_hop_first(
                rng,
                num_conflict_rows=num_conflict_rows,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "passkey_copy":
            payload = generate_passkey_copy_hop_first(
                rng,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "passkey_distractor":
            payload = generate_passkey_distractor_hop_first(
                rng,
                num_distractors=num_distractors,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "addr_val_active":
            payload = generate_addr_val_active_hop_first(rng, num_distractors=num_distractors)
        elif task_type == "ptr_chain":
            payload = generate_ptr_chain_hop_first(
                rng,
                hop_count=hop_fixed,
                num_distractors=num_decoy_addrs,
            )
        elif task_type == "massive_addr_val":
            payload = generate_massive_addr_val_hop_first(
                rng,
                num_kv_pairs=num_kv_pairs,
                answer_digit_width=answer_digit_width,
            )
        elif task_type == "mqar_addr_val":
            payload = generate_mqar_addr_val_hop_first(
                rng,
                num_kv_pairs=num_kv_pairs,
                answer_digit_width=answer_digit_width,
                num_queries=num_queries,
                supervise_all_queries=mqar_supervise_all,
            )
        else:
            payload = SYNTHETIC_TASK_GENERATORS[task_type](rng)
        try:
            enforce_max_answer_chars(payload, max_answer)
        except ValueError:
            continue
        if question_leaks_answer(payload):
            continue
        return payload
    raise RuntimeError(f"Failed to generate synthetic task {task_type!r}")
