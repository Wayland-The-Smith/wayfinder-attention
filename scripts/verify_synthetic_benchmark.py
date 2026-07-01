#!/usr/bin/env python3
"""Sanity tests for synthetic L0–L4 pointer / address NIAH benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context import (
    LongContextBenchmarkConfig,
    LongContextSampleGenerator,
)
from routing_attention.benchmarks.long_context.dataset import LongContextTrainDataset
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import (
    _filler_text,
    _is_causally_reachable,
    _needle_spans,
    _suffix_start_forbidden,
)
from routing_attention.benchmarks.long_context.holdout import clear_holdout_cache, get_holdout_grid
from routing_attention.benchmarks.long_context.tasks_synthetic import (
    SYNTHETIC_ALL_TASK_TYPES,
    SYNTHETIC_PRIMARY_GATE_TASK_TYPES,
    SYNTHETIC_SECONDARY_TASK_TYPES,
    SYNTHETIC_TASK_TYPES,
    ACTIVE_TAG,
    ADDR_TAG,
    PTR_TAG,
    VAL_TAG,
    _ACTIVE_TAG_KEY_COLLISIONS,
    generate_synthetic_task,
)


def _cfg() -> LongContextBenchmarkConfig:
    return LongContextBenchmarkConfig().apply_synthetic_profile()


def _haystack_text(gen, sample) -> str:
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    suffix = (
        f"{gen.config.question_prefix}{sample.question}"
        f"{gen.config.answer_prefix}{sample.expected_answer}"
    )
    ss = int(sample.meta_dict["suffix_start"])
    return text[:ss] + text[ss + len(suffix) :]


def test_all_synthetic_tasks_generate():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    for task in SYNTHETIC_ALL_TASK_TYPES:
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type=task,
            haystack_mode="synthetic_noise",
            seed=hash(task) % 100000,
        )
        assert sample.expected_answer
        assert sample.answer_end > sample.answer_start
    print(f"all synthetic tasks generate OK ({len(SYNTHETIC_ALL_TASK_TYPES)})")


def test_l0_unique_key_once():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="pointer_unique",
        seed=101,
    )
    key = sample.metadata["key"]
    haystack = _haystack_text(gen, sample)
    assert haystack.count(f"{key} ") == 1, f"L0: key {key!r} must appear once in haystack"
    assert sample.metadata.get("num_distractors", 0) >= 1, "L0 needs decoy rows"
    assert sample.expected_answer in haystack
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    assert f"Q {key}" in text
    assert f"Q {key} A {sample.expected_answer}" in text
    print("L0 unique key OK")


def test_l1_single_active_row():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="pointer_active",
        seed=202,
    )
    key = sample.metadata["key"]
    haystack = _haystack_text(gen, sample)
    assert haystack.count(f"{ACTIVE_TAG} {key} ") == 1
    assert f"{ACTIVE_TAG} {key} {sample.expected_answer}" in haystack
    assert haystack.count(f"{key} ") >= 3
    print("L1 single ACTIVE row OK")


def test_l1_many_distractors():
    """L1 must support >9 distractors (wider fake values) without hanging."""
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        task_types=["pointer_active"],
        needle_depths=[0.5],
        suffix_placement="at_end",
        num_distractors=14,
        benchmark_family="synthetic",
    ).apply_synthetic_profile()
    gen = LongContextSampleGenerator(cfg)
    for seed in range(30):
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type="pointer_active",
            seed=7000 + seed,
        )
        assert sample.metadata["num_distractors"] == 14
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        q = int(sample.meta_dict["question_start"])
        assert _is_causally_reachable(
            text,
            q,
            sample.metadata["key"],
            sample.metadata["value"],
            gold_pattern=sample.metadata["gold_needle"],
        )
    print("L1 many distractors OK (30 samples @ 14 decoys)")


def test_l2_addr_val_lookup():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="addr_val",
        seed=303,
    )
    addr = sample.metadata["addr"]
    haystack = _haystack_text(gen, sample)
    assert f"{ADDR_TAG} {addr} {VAL_TAG} {sample.expected_answer}" in haystack
    assert haystack.count(f"{ADDR_TAG} ") >= 2, "L2 needs decoy ADDR rows"
    assert sample.question == f"QUERY {addr}"
    print("L2 addr_val OK")


def test_l3_single_active_addr():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="addr_val_active",
        seed=404,
    )
    addr = sample.metadata["addr"]
    haystack = _haystack_text(gen, sample)
    assert f"{ACTIVE_TAG} {ADDR_TAG} {addr} {VAL_TAG} {sample.expected_answer}" in haystack
    assert haystack.count(f"{ADDR_TAG} {addr} ") >= 3
    print("L3 addr_val_active OK")


def test_l4_ptr_chain_consistency():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="ptr_chain",
        seed=505,
    )
    chain = sample.metadata["chain"]
    value = sample.metadata["value"]
    haystack = _haystack_text(gen, sample)
    assert len(chain) >= 2
    for i in range(len(chain) - 1):
        hop = f"{ADDR_TAG} {chain[i]} {PTR_TAG} {chain[i + 1]}."
        assert hop in haystack, f"missing hop segment {hop!r}"
    terminal = f"{ADDR_TAG} {chain[-1]} {VAL_TAG} {value}."
    assert terminal in haystack
    assert haystack.count(f"{VAL_TAG} ") >= 2, "L4 needs decoy VAL rows"
    assert haystack.count(f"{ADDR_TAG} {chain[0]} {PTR_TAG} ") == 1, "L4 gold PTR from start is unique"
    assert sample.metadata.get("protocol_version") == 2
    assert sample.expected_answer == str(value)
    assert sample.question == f"QUERY {chain[0]}"
    print("L4 ptr_chain OK")


def test_ptr_chain_fixed_hop_count():
    """When min==max, every sample must share the same chain length."""
    import random

    rng = random.Random(42)
    for fixed in (2, 3, 4):
        counts: list[int] = []
        for _ in range(24):
            p = generate_synthetic_task(
                "ptr_chain",
                rng,
                {
                    "synthetic_hop_count": fixed,
                    "synthetic_hop_count_min": fixed,
                    "synthetic_hop_count_max": fixed,
                },
            )
            counts.append(int(p.metadata["hop_count"]))
            assert int(p.metadata["ptr_hops"]) == fixed - 1
        assert len(set(counts)) == 1 and counts[0] == fixed
    print("ptr_chain fixed hop_count OK")


def test_answer_only_supervision():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    for task in SYNTHETIC_PRIMARY_GATE_TASK_TYPES:
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type=task,
            seed=999,
        )
        labels = sample.labels.numpy()
        a0, a1 = sample.answer_start, sample.answer_end
        assert (labels[:a0] == -100).all()
        assert (labels[a0:a1] != -100).all()
    print("answer-only supervision OK")


def test_no_question_leak():
    import random

    rng = random.Random(0)
    for task in SYNTHETIC_ALL_TASK_TYPES:
        for _ in range(20):
            p = generate_synthetic_task(task, rng)
            assert p.expected_answer.lower() not in p.question.lower()
    print("no question leak OK")


def test_training_includes_all_levels():
    cfg = _cfg()
    ds = LongContextTrainDataset(cfg, batch_size=1, train_context_length=512)
    assert set(ds._tasks) == set(SYNTHETIC_TASK_TYPES)
    print("training includes all L0–L4 OK")


def test_holdout_same_tasks_disjoint_seed():
    cfg = _cfg()
    assert cfg.holdout_seed != cfg.seed
    holdout_cfg = cfg.holdout_config()
    assert holdout_cfg.seed == cfg.holdout_seed
    cfg = LongContextBenchmarkConfig(
        **{**cfg.to_dict(), "context_lengths": [512], "eval_samples_per_cell": 1}
    )
    clear_holdout_cache()
    holdout = get_holdout_grid(cfg)
    tasks = {s.task_type for s in holdout}
    assert tasks == set(SYNTHETIC_TASK_TYPES)
    gen = LongContextSampleGenerator(cfg)
    train = gen.generate_one(
        context_length=512,
        needle_depth=0.5,
        task_type="pointer_unique",
        seed=cfg.seed,
    )
    hold = next(s for s in holdout if s.task_type == "pointer_unique")
    assert train.input_ids.tolist() != hold.input_ids.tolist()
    print("holdout same tasks, disjoint samples OK")


def test_suffix_never_splits_needles():
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    for task in SYNTHETIC_ALL_TASK_TYPES:
        sample = gen.generate_one(
            context_length=4096,
            needle_depth=0.5,
            task_type=task,
            seed=707,
        )
        ss = int(sample.meta_dict["suffix_start"])
        segments = sample.meta_dict.get("needle_segments", [])
        haystack = _haystack_text(gen, sample)
        forbidden = _suffix_start_forbidden(_needle_spans(haystack, segments))
        assert ss not in forbidden
    print("suffix never splits needles OK")


def test_at_end_suffix_placement():
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        task_types=["pointer_unique"],
        suffix_placement="at_end",
    ).apply_synthetic_profile()
    gen = LongContextSampleGenerator(cfg)
    for depth in (0.1, 0.5, 0.9):
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=depth,
            task_type="pointer_unique",
            seed=808 + int(depth * 100),
        )
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        assert text.rstrip().endswith(sample.expected_answer)
        assert int(sample.meta_dict["suffix_start"]) == len(_haystack_text(gen, sample))
        q = int(sample.meta_dict["question_start"])
        assert _is_causally_reachable(
            text, q, sample.metadata["key"], sample.metadata["value"]
        )
    print("at_end suffix placement OK")


def test_random_suffix_causally_reachable():
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        task_types=["pointer_unique"],
        suffix_placement="random",
        suffix_depth_min=0.1,
        suffix_depth_max=0.9,
        needle_depths=[0.1, 0.25, 0.5, 0.75, 0.9],
    ).apply_synthetic_profile()
    gen = LongContextSampleGenerator(cfg)
    for seed in range(200):
        depth = [0.1, 0.25, 0.5, 0.75, 0.9][seed % 5]
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=depth,
            task_type="pointer_unique",
            seed=900 + seed,
        )
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        q = int(sample.meta_dict["question_start"])
        assert _is_causally_reachable(
            text, q, sample.metadata["key"], sample.metadata["value"]
        ), f"seed={seed} depth={depth} unreachable"
    print("random suffix causal reachability OK (200 samples)")


def _sample_haystack_and_filler(gen, sample):
    haystack = _haystack_text(gen, sample)
    segments = sample.meta_dict.get("needle_segments", [])
    filler = _filler_text(haystack, segments)
    return haystack, segments, filler


def test_filler_never_mimics_needles():
    """Filler must not contain structured tokens, uppercase letters, or digits."""
    import re

    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    tags = ("ACTIVE", "ADDR", "VAL", "PTR")
    for task in SYNTHETIC_ALL_TASK_TYPES:
        for seed in range(100):
            sample = gen.generate_one(
                context_length=4096,
                needle_depth=0.5,
                task_type=task,
                seed=1000 + seed,
            )
            haystack, segments, filler = _sample_haystack_and_filler(gen, sample)
            for tag in tags:
                assert tag not in filler, f"{task} seed={seed}: filler contains {tag!r}"
            assert not re.search(r"[A-Za-z0-9]", filler), (
                f"{task} seed={seed}: filler has alphanumeric chars: {filler[:80]!r}"
            )
            for seg in segments:
                assert haystack.count(seg) == 1, f"{task}: segment {seg!r} count != 1"
    print("filler never mimics needles OK (500 samples)")


def test_intentional_needle_counts():
    """Structured rows in the haystack match exactly what each level intends."""
    cfg = _cfg()
    gen = LongContextSampleGenerator(cfg)
    for seed in range(30):
        # L0
        s = gen.generate_one(
            context_length=4096, needle_depth=0.5, task_type="pointer_unique", seed=2000 + seed
        )
        haystack, _, _ = _sample_haystack_and_filler(gen, s)
        key = s.metadata["key"]
        assert haystack.count(f"{key} ") == 1
        assert s.metadata["num_distractors"] >= 1

        # L1 — count rows, not ``{key} `` substrings (``E`` appears inside ``ACTIVE``).
        s = gen.generate_one(
            context_length=4096, needle_depth=0.5, task_type="pointer_active", seed=3000 + seed
        )
        haystack, segments, _ = _sample_haystack_and_filler(gen, s)
        key = s.metadata["key"]
        assert key not in _ACTIVE_TAG_KEY_COLLISIONS
        assert len(segments) == s.metadata["num_distractors"] + 1
        assert sum(1 for seg in segments if seg.startswith(f"{ACTIVE_TAG} {key} ")) == 1
        assert sum(1 for seg in segments if seg.startswith(f"{key} ")) == s.metadata["num_distractors"]

        # L2
        s = gen.generate_one(
            context_length=4096, needle_depth=0.5, task_type="addr_val", seed=4000 + seed
        )
        haystack, _, _ = _sample_haystack_and_filler(gen, s)
        addr = s.metadata["addr"]
        assert haystack.count(f"{ADDR_TAG} {addr} {VAL_TAG} ") == 1
        assert haystack.count(f"{ADDR_TAG} ") == s.metadata["num_distractors"] + 1

        # L3
        s = gen.generate_one(
            context_length=4096, needle_depth=0.5, task_type="addr_val_active", seed=5000 + seed
        )
        haystack, _, _ = _sample_haystack_and_filler(gen, s)
        addr = s.metadata["addr"]
        assert haystack.count(f"{ACTIVE_TAG} {ADDR_TAG} {addr} ") == 1
        assert haystack.count(f"{ADDR_TAG} {addr} ") == s.metadata["num_distractors"] + 1

        # L4
        s = gen.generate_one(
            context_length=4096, needle_depth=0.5, task_type="ptr_chain", seed=6000 + seed
        )
        haystack, _, _ = _sample_haystack_and_filler(gen, s)
        chain = s.metadata["chain"]
        value = s.metadata["value"]
        terminal = f"{ADDR_TAG} {chain[-1]} {VAL_TAG} {value}."
        assert haystack.count(terminal) == 1
        assert haystack.count(f"{ADDR_TAG} {chain[0]} {PTR_TAG} ") == 1
        for wrong_val_seg in haystack.split("."):
            if f"{VAL_TAG} {value}" in wrong_val_seg and terminal.rstrip(".") not in wrong_val_seg:
                raise AssertionError(f"L4: answer value {value!r} on non-terminal row")
    print("intentional needle counts OK (30 seeds x 5 levels)")


def test_l4_decoy_vals_differ_from_answer():
    import random

    rng = random.Random(0)
    for _ in range(100):
        p = generate_synthetic_task("ptr_chain", rng, {"synthetic_hop_count": 3})
        value = p.metadata["value"]
        terminal_addr = str(p.metadata["chain"][-1])
        for seg in p.needle_segments:
            if f"{VAL_TAG} {value}" in seg and terminal_addr not in seg:
                raise AssertionError(f"decoy VAL repeats answer: {seg!r}")
    print("L4 decoy VALs differ from answer OK")


def test_primary_gate_metric():
    cfg = _cfg()
    ev = LongContextEvaluator(cfg)
    from routing_attention.benchmarks.long_context.evaluation import EvalRecord

    records = [
        EvalRecord(True, "3", "3", "addr_val", 512, 0.5),
        EvalRecord(False, "1", "2", "pointer_unique", 512, 0.5),
        EvalRecord(True, "9", "9", "ptr_chain", 512, 0.5),
    ]
    summary = ev.summarize(records)
    assert summary.primary_gate_total == 3
    assert summary.primary_gate_correct == 2
    assert summary.secondary_total == 0
    print("primary gate metric OK")


def main():
    test_all_synthetic_tasks_generate()
    test_l0_unique_key_once()
    test_l1_single_active_row()
    test_l1_many_distractors()
    test_l2_addr_val_lookup()
    test_l3_single_active_addr()
    test_l4_ptr_chain_consistency()
    test_ptr_chain_fixed_hop_count()
    test_no_question_leak()
    test_training_includes_all_levels()
    test_holdout_same_tasks_disjoint_seed()
    test_suffix_never_splits_needles()
    test_at_end_suffix_placement()
    test_random_suffix_causally_reachable()
    test_filler_never_mimics_needles()
    test_intentional_needle_counts()
    test_l4_decoy_vals_differ_from_answer()
    test_primary_gate_metric()
    print("\nAll synthetic NIAH benchmark checks passed.")


if __name__ == "__main__":
    main()
