"""Task generators for long-context retrieval benchmark (types A–F)."""



from __future__ import annotations



import random
import re
import string

from dataclasses import dataclass





@dataclass

class TaskPayload:

    """Needle text segments, question, and expected answer."""



    needle_segments: list[str]

    question: str

    expected_answer: str

    task_type: str

    metadata: dict

    # Per-character mask within expected_answer (True = supervise). None = all chars.

    answer_supervision_mask: tuple[bool, ...] | None = None



    def collision_check_strings(self) -> list[str]:

        """Substrings that must not appear in haystack filler (false-positive guard)."""

        checks = list(self.metadata.get("collision_checks", []))

        ans = self.expected_answer.strip()

        if ans and ans not in checks:

            checks.append(ans)

        return checks





def _magic_number(rng: random.Random) -> str:

    return str(rng.randint(10000, 99999))





def _region_code(rng: random.Random, length: int = 8) -> str:

    alphabet = string.ascii_uppercase + string.digits

    return "".join(rng.choice(alphabet) for _ in range(length))





def _attribute_code(rng: random.Random) -> str:

    return str(rng.randint(100000, 999999))





def _supervision_mask_for_answer(answer: str, supervise_indices: set[int] | None = None) -> tuple[bool, ...]:

    if supervise_indices is None:

        return tuple(True for _ in answer)

    return tuple(i in supervise_indices for i in range(len(answer)))





_MAGIC_NUMBER_QUESTIONS = (

    "What is the magic number?",

    "Which magic number is stated in the passage?",

    "Report the magic number from the document.",

    "What magic number appears in the text?",

    "State the magic number recorded above.",

    "Find the magic number in the context.",

)



_MAGIC_NEEDLE_TEMPLATES = (

    "The magic number is {value}.",

    "Recorded magic number: {value}.",

    "Magic number entry = {value}.",

    "Passage lists magic number {value}.",

    "Note: magic number {value}.",

)





def _magic_number_question(rng: random.Random) -> str:

    return rng.choice(_MAGIC_NUMBER_QUESTIONS)





def _magic_needle(rng: random.Random, value: str) -> str:

    return rng.choice(_MAGIC_NEEDLE_TEMPLATES).format(value=value)





_KEY_VALUE_QUESTION_TEMPLATES = (

    "What is the balance of UserID {uid}?",

    "Report the balance for UserID {uid}.",

    "How much balance does UserID {uid} have?",

    "What balance is listed for UserID {uid}?",

)



_KEY_VALUE_NEEDLE_TEMPLATES = (

    "UserID {uid} has balance {balance}.",

    "Account UserID {uid} balance {balance}.",

    "UserID {uid} balance recorded as {balance}.",

    "Ledger entry UserID {uid} = {balance}.",

)





def _person_name(rng: random.Random) -> str:

    first = rng.choice(

        (

            "Alice",

            "Bob",

            "Charlie",

            "Diana",

            "Eve",

            "Frank",

            "Grace",

            "Henry",

            "Iris",

            "Jack",

            "Karen",

            "Leo",

        )

    )

    suffix = rng.randint(100, 999)

    return f"{first}{suffix}"





_MULTI_KEY_SEGMENT_TEMPLATES = (

    "{name} maps to {code}.",

    "Entry {name} -> {code}.",

    "Registry: {name} = {code}.",

)



_MULTI_HOP_SEG1_TEMPLATES = (

    "{person} lives in {city}.",

    "Resident {person} is based in {city}.",

)



_MULTI_HOP_SEG2_TEMPLATES = (

    "{city} is located in region {region}.",

    "Region code for {city} is {region}.",

    "{city} belongs to region {region}.",

)



_VAULT_NEEDLE_TEMPLATES = (

    "{vault_id} access code {code} is active.",

    "Vault {vault_id} uses access code {code}.",

    "Active code for {vault_id}: {code}.",

)





def generate_exact_retrieval(rng: random.Random) -> TaskPayload:

    value = _magic_number(rng)

    needle = _magic_needle(rng, value)

    return TaskPayload(

        needle_segments=[needle],

        question=_magic_number_question(rng),

        expected_answer=value,

        task_type="exact_retrieval",

        metadata={"needle_kind": "magic_number", "collision_checks": [value]},

    )





def generate_key_value(rng: random.Random) -> TaskPayload:

    uid = rng.randint(1000, 9999)

    balance = rng.randint(100000, 999999)

    needle = rng.choice(_KEY_VALUE_NEEDLE_TEMPLATES).format(uid=uid, balance=balance)

    question = rng.choice(_KEY_VALUE_QUESTION_TEMPLATES).format(uid=uid)

    answer = str(balance)

    return TaskPayload(

        needle_segments=[needle],

        question=question,

        expected_answer=answer,

        task_type="key_value",

        metadata={"user_id": uid, "balance": balance, "collision_checks": [answer]},

    )





def generate_multi_key(rng: random.Random, num_keys: int = 3) -> TaskPayload:

    n = max(2, num_keys)

    names = [_person_name(rng) for _ in range(n)]

    codes = [_attribute_code(rng) for _ in range(n)]

    segments = [

        rng.choice(_MULTI_KEY_SEGMENT_TEMPLATES).format(name=name, code=code)

        for name, code in zip(names, codes)

    ]

    target_idx = rng.randint(0, n - 1)

    target = names[target_idx]

    answer = codes[target_idx]

    return TaskPayload(

        needle_segments=segments,

        question=f"What code is assigned to {target}?",

        expected_answer=answer,

        task_type="multi_key",

        metadata={

            "target": target,

            "pairs": list(zip(names, codes)),

            "collision_checks": list(codes),

        },

    )





def generate_multi_hop(rng: random.Random) -> TaskPayload:

    person = _person_name(rng)

    city = (

        f"{rng.choice(('North', 'South', 'East', 'West', 'Lake', 'Mount'))}"

        f"{rng.choice(('ville', 'ton', 'ford', 'dale', 'field'))}{rng.randint(10, 99)}"

    )

    region = _region_code(rng)

    seg1 = rng.choice(_MULTI_HOP_SEG1_TEMPLATES).format(person=person, city=city)

    seg2 = rng.choice(_MULTI_HOP_SEG2_TEMPLATES).format(city=city, region=region)

    return TaskPayload(

        needle_segments=[seg1, seg2],

        question=f"Which region does {person} live in?",

        expected_answer=region,

        task_type="multi_hop",

        metadata={"person": person, "city": city, "region": region, "collision_checks": [region]},

    )





def generate_distractor(rng: random.Random, num_distractors: int = 8) -> TaskPayload:

    target = _magic_number(rng)

    distractors = {_magic_number(rng) for _ in range(num_distractors)}

    distractors.discard(target)

    segments = [_magic_needle(rng, d) for d in sorted(distractors)]

    insert_at = rng.randint(0, len(segments))

    segments.insert(insert_at, _magic_needle(rng, target))

    all_values = [target] + sorted(distractors)

    return TaskPayload(

        needle_segments=segments,

        question=_magic_number_question(rng),

        expected_answer=target,

        task_type="distractor",

        metadata={

            "target": target,

            "num_distractors": len(distractors),

            "collision_checks": all_values,

        },

    )





def generate_multiple_needles(rng: random.Random, num_needles: int = 3) -> TaskPayload:

    """Multi-needle haystack with single-vault retrieval (pure NIAH — no list aggregation)."""

    n = max(2, num_needles)

    codes = [rng.randint(100000, 999999) for _ in range(n)]

    vault_ids = [

        f"Vault{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}" for _ in range(n)

    ]

    segments = [

        rng.choice(_VAULT_NEEDLE_TEMPLATES).format(vault_id=vault_id, code=code)

        for vault_id, code in zip(vault_ids, codes)

    ]

    target_idx = rng.randint(0, n - 1)

    question = f"What is the access code for {vault_ids[target_idx]}?"

    answer = str(codes[target_idx])

    return TaskPayload(

        needle_segments=segments,

        question=question,

        expected_answer=answer,

        task_type="multiple_needles",

        metadata={

            "codes": codes,

            "vault_ids": vault_ids,

            "target_vault": vault_ids[target_idx],

            "target_code": codes[target_idx],

            "collision_checks": [str(c) for c in codes],

        },

    )





_MAGIC_VALUE_RE = re.compile(r"magic number[^0-9]*(\d{5,6})", re.IGNORECASE)


def trace_exact_retrieval(text: str) -> str | None:
    """Return the sole magic-number value in ``text`` (NL passkey copy)."""
    hits = _MAGIC_VALUE_RE.findall(text)
    if len(hits) != 1:
        return None
    return hits[0]


def trace_distractor(text: str, target: str) -> str | None:
    """Target magic number must appear in exactly one needle."""
    hits = _MAGIC_VALUE_RE.findall(text)
    if hits.count(target) != 1:
        return None
    return target


def trace_nl_task_answer(text: str, payload: TaskPayload) -> str | None:
    if payload.task_type == "exact_retrieval":
        return trace_exact_retrieval(text)
    if payload.task_type == "distractor":
        return trace_distractor(text, str(payload.metadata.get("target", payload.expected_answer)))
    return None


def question_leaks_answer(payload: TaskPayload) -> bool:

    """True when the expected answer appears verbatim in the question text."""

    q = payload.question.strip().lower()

    a = payload.expected_answer.strip().lower()

    if not a:

        return False

    # Single-token numeric answers (addr_val / mqar): avoid false positives when a
    # digit appears only inside a multi-digit address (e.g. answer "1" in "QUERY 91").
    if a.isdigit() and len(a) <= 3:

        return a in q.split()

    return a in q





def enforce_max_answer_chars(payload: TaskPayload, max_chars: int) -> None:

    if len(payload.expected_answer) > max_chars:

        raise ValueError(

            f"Task {payload.task_type!r} answer length {len(payload.expected_answer)} "

            f"exceeds max_answer_chars={max_chars}"

        )





TASK_GENERATORS = {

    "exact_retrieval": generate_exact_retrieval,

    "key_value": generate_key_value,

    "multi_key": lambda rng: generate_multi_key(rng, num_keys=3),

    "multi_hop": generate_multi_hop,

    "distractor": lambda rng: generate_distractor(rng, num_distractors=8),

    "multiple_needles": lambda rng: generate_multiple_needles(rng, num_needles=3),

}



# Primary go/no-go gate: semantically valid single-key NIAH retrieval.

PRIMARY_GATE_TASK_TYPES = (

    "exact_retrieval",

    "key_value",

    "multiple_needles",

)

# Monitored in holdout only — ambiguous multi-needle design; excluded from training/gate.

SECONDARY_TASK_TYPES = (

    "distractor",

)

# Legacy aliases (prefer PRIMARY_GATE_TASK_TYPES).

PURE_NIAH_TASK_TYPES = PRIMARY_GATE_TASK_TYPES

RETRIEVAL_HEAVY_TASK_TYPES = PRIMARY_GATE_TASK_TYPES





def generate_task(task_type: str, rng: random.Random, config_kwargs: dict | None = None) -> TaskPayload:

    if task_type not in TASK_GENERATORS:

        raise ValueError(f"Unknown task type '{task_type}'. Available: {list(TASK_GENERATORS)}")

    cfg = config_kwargs or {}

    max_answer = int(cfg.get("max_answer_chars", 64))

    for _ in range(32):

        if task_type == "multi_key":

            payload = generate_multi_key(rng, num_keys=int(cfg.get("num_multi_keys", 3)))

        elif task_type == "distractor":

            payload = generate_distractor(rng, num_distractors=int(cfg.get("num_distractors", 8)))

        elif task_type == "multiple_needles":

            payload = generate_multiple_needles(rng, num_needles=int(cfg.get("num_needles_multi", 3)))

        else:

            payload = TASK_GENERATORS[task_type](rng)

        try:

            enforce_max_answer_chars(payload, max_answer)

        except ValueError:

            continue

        if question_leaks_answer(payload):

            continue

        return payload

    raise RuntimeError(

        f"Failed to generate valid task {task_type!r} within max_answer_chars={max_answer}"

    )


