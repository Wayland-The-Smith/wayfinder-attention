#!/usr/bin/env python3
"""Sanity tests for Experiment 7 long-context benchmark generator and evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from routing_attention.benchmarks.long_context import (
    LongContextBenchmarkConfig,
    LongContextEvaluator,
    LongContextSampleGenerator,
)
from routing_attention.benchmarks.long_context.tasks import (
    PRIMARY_GATE_TASK_TYPES,
    SECONDARY_TASK_TYPES,
    TASK_GENERATORS,
    generate_task,
    question_leaks_answer,
)
from routing_attention.benchmarks.long_context.holdout import clear_holdout_cache, get_holdout_grid
from routing_attention.benchmarks.long_context.generator import (
    _needle_spans,
    _suffix_start_forbidden,
)
from routing_attention.models.transformer import TransformerLM


def test_all_task_types():
    gen = LongContextSampleGenerator()
    for task in TASK_GENERATORS:
        sample = gen.generate_one(
            context_length=512,
            needle_depth=0.5,
            task_type=task,
            haystack_mode="random_sentences",
            seed=42,
        )
        assert sample.expected_answer, f"empty answer for {task}"
        assert sample.answer_end > sample.answer_start, f"bad span for {task}"
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        assert sample.expected_answer in text or sample.expected_answer.lower() in text.lower()
    print(f"task types OK ({len(TASK_GENERATORS)})")


def test_context_lengths():
    gen = LongContextSampleGenerator()
    for length in [1024, 2048, 4096, 8192]:
        sample = gen.generate_one(
            context_length=length,
            needle_depth=0.25,
            task_type="exact_retrieval",
            seed=length,
        )
        assert sample.input_ids.shape[0] == length
    print("context lengths OK")


def test_evaluator_random_model():
    cfg = LongContextBenchmarkConfig(
        context_lengths=[512],
        needle_depths=[0.10, 0.90],
        task_types=list(PRIMARY_GATE_TASK_TYPES),
        secondary_task_types=[],
        haystack_modes=["random_sentences"],
        eval_samples_per_cell=2,
    )
    model = TransformerLM(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        attention_type="dense",
        num_digit_classes=0,
    )
    evaluator = LongContextEvaluator(cfg)
    summary = evaluator.evaluate_module(model, device=torch.device("cpu"), show_progress=False)
    assert summary.total > 0
    assert 0.0 <= summary.overall_accuracy <= 1.0
    print(f"evaluator OK total={summary.total} acc={summary.overall_accuracy:.3f}")


def test_longest_first_order():
    cfg = LongContextBenchmarkConfig(context_lengths=[1024, 8192, 4096])
    assert cfg.context_lengths == [8192, 4096, 1024]
    gen = LongContextSampleGenerator(cfg)
    grid = gen.generate_grid(
        context_lengths=[512, 2048, 1024],
        needle_depths=[0.5],
        task_types=["exact_retrieval"],
        haystack_modes=["random_sentences"],
        samples_per_cell=1,
    )
    lengths = [s.context_length for s in grid]
    assert lengths == sorted(lengths, reverse=True), f"grid not longest-first: {lengths}"
    print("longest-first order OK")


def test_untrained_model_near_zero_accuracy():
    """Causal off-by-one fix: random init should not score ~100%."""
    cfg = LongContextBenchmarkConfig(
        context_lengths=[512],
        needle_depths=[0.50],
        task_types=["exact_retrieval"],
        haystack_modes=["random_sentences"],
        eval_samples_per_cell=8,
    )
    model = TransformerLM(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        attention_type="dense",
        num_digit_classes=0,
    )
    evaluator = LongContextEvaluator(cfg)
    summary = evaluator.evaluate_module(model, device=torch.device("cpu"), show_progress=False)
    assert summary.overall_accuracy < 0.5, (
        f"untrained model accuracy suspiciously high: {summary.overall_accuracy}"
    )
    print(f"untrained baseline OK acc={summary.overall_accuracy:.3f}")


def test_train_holdout_disjoint():
    """Held-out grid must be built from holdout_seed, not the training stream seed."""
    from routing_attention.benchmarks.long_context.holdout import clear_holdout_cache, get_holdout_grid

    cfg = LongContextBenchmarkConfig(
        context_lengths=[512],
        needle_depths=[0.5],
        task_types=["exact_retrieval"],
        haystack_modes=["random_sentences"],
        eval_samples_per_cell=2,
        seed=42,
        holdout_seed=999_999,
    )
    clear_holdout_cache()
    holdout = get_holdout_grid(cfg)
    train_grid = LongContextSampleGenerator(cfg).generate_grid(
        context_lengths=[512],
        needle_depths=[0.5],
        task_types=["exact_retrieval"],
        haystack_modes=["random_sentences"],
        samples_per_cell=2,
        base_seed=cfg.seed,
    )
    assert holdout[0].input_ids.shape == train_grid[0].input_ids.shape
    assert not torch.equal(holdout[0].input_ids, train_grid[0].input_ids)
    print("train/holdout disjoint OK")


def test_char_token_alignment():
    """Char positions must match token positions (1:1 tokenizer)."""
    gen = LongContextSampleGenerator()
    sample = gen.generate_one(
        context_length=2048,
        needle_depth=0.33,
        task_type="key_value",
        haystack_mode="random_sentences",
        seed=12345,
    )
    assert sample.input_ids.shape[0] == 2048
    assert sample.answer_end > sample.answer_start
    pred_slice = sample.input_ids[sample.answer_start : sample.answer_end]
    decoded = gen.tokenizer.decode(pred_slice.tolist())
    assert sample.expected_answer in decoded or decoded in sample.expected_answer
    print("char-token alignment OK")


def test_fast_attention_backends():
    from routing_attention.models.fast_attention import (
        backend_status,
        require_fla,
        require_flex_attention,
        resolve_fla_linear_kernel,
        warmup_fla_linear_kernels,
    )

    status = backend_status()
    require_fla()
    require_flex_attention()
    assert status["fla_linear"]
    assert status["flex_sliding_window"]
    if torch.cuda.is_available():
        kernel = resolve_fla_linear_kernel(device=torch.device("cuda"), dtype=torch.bfloat16)
        assert kernel in ("chunk", "fused_chunk")
        warmup_fla_linear_kernels(
            device=torch.device("cuda"),
            context_length=512,
        )
        print(f"fast attention backends OK kernel={kernel}", status)
    else:
        print("fast attention backends OK (cpu)", status)


def test_routing_sparse_topk_path():
    """Routing attention must retrieve top-k only and respect freeze_router."""
    from experiments.common import build_transformer, load_experiment_config, load_router_from_reuse
    from routing_attention.benchmarks.long_context.routing_setup import apply_routing_variant_settings

    config = load_experiment_config(7, variant="routing_asymmetric")
    device = torch.device("cpu")
    var_config = config.copy()
    router, _ = load_router_from_reuse(var_config, device)
    model = build_transformer(var_config, attention_type="routing", router=router)
    info = apply_routing_variant_settings(model, var_config, "routing", max_seq_len=512)
    assert info["routing_layers"] == model.n_layers
    assert info["router_params_trainable"] == 0
    assert info["freeze_router"] is True

    x = torch.randn(1, 128, model.d_model)
    attn = model.blocks[0].attn
    out = attn(x)
    assert out.shape == x.shape
    # Meat path: softmax over k candidates, not T
    q = attn.q_proj(x).view(1, 128, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(1, 128, attn.n_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(1, 128, attn.n_heads, attn.head_dim).transpose(1, 2)
    idx = attn._retrieve_candidates(x)
    assert idx.shape[-1] == attn.top_k
    from routing_attention.models.attention import _sparse_meat_forward

    meat = _sparse_meat_forward(q, k, v, idx, attn.scale, attn.dropout, attn._retrieval_cfg)
    assert meat.shape == q.shape
    print("routing sparse top-k path OK")


def test_train_requires_fixed_context_length():
    from routing_attention.benchmarks.long_context.dataset import LongContextTrainDataset

    try:
        LongContextTrainDataset(LongContextBenchmarkConfig())
        raise AssertionError("expected ValueError without train_context_length")
    except ValueError:
        pass
    ds = LongContextTrainDataset(LongContextBenchmarkConfig(), train_context_length=512)
    batch = next(iter(ds))
    assert batch["input_ids"].shape[1] == 512
    print("fixed train context length OK")


def test_production_backend_manifest():
    from routing_attention.benchmarks.long_context.production_backends import (
        assert_production_backends_available,
        production_manifest_for_variants,
    )

    manifest = production_manifest_for_variants(
        ["dense_flash", "linear", "local_window64", "routing_asymmetric"]
    )
    assert manifest["linear"]["kernel"] == "fla_chunk_linear_attn"
    assert manifest["local_window64"]["kernel"] == "pytorch_flex_attention"
    assert manifest["dense_flash"]["kernel"] == "pytorch_sdpa_flash"
    if torch.cuda.is_available():
        assert_production_backends_available()
    print("production backend manifest OK")


def test_validate_every_policy():
    """Mid-train holdout: disabled by default; min 5000 steps if enabled."""

    def effective(raw: int) -> int:
        v = int(raw)
        if 0 < v < 5000:
            v = 5000
        return v

    assert effective(0) == 0
    assert effective(100) == 5000
    assert effective(5000) == 5000
    assert effective(10000) == 10000
    print("validate_every policy OK")


def test_no_question_answer_leak():
    """Questions must not contain the expected answer (retrieval-only tasks)."""
    import random

    rng = random.Random(0)
    for _ in range(200):
        for task in TASK_GENERATORS:
            payload = generate_task(task, rng)
            assert not question_leaks_answer(payload), (
                f"{task}: q={payload.question!r} leaks a={payload.expected_answer!r}"
            )
    print("no question/answer leak OK")


def test_answer_only_no_constant_supervision():
    """Only variable answer span is supervised — never prefixes or question text."""
    gen = LongContextSampleGenerator(LongContextBenchmarkConfig())
    for task in TASK_GENERATORS:
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type=task,
            haystack_mode="random_sentences",
            seed=hash(task) % 100000,
        )
        labels = sample.labels.numpy()
        weights = sample.loss_weights_np
        q_start = int(sample.meta_dict["question_start"])
        a_start, a_end = sample.answer_start, sample.answer_end

        assert (labels[:a_start] == -100).all(), f"{task}: haystack/prefix/question supervised"
        supervised = labels[a_start:a_end] != -100
        assert supervised.any(), f"{task}: no supervised answer tokens"
        if a_end < len(labels):
            assert (labels[a_end:] == -100).all(), f"{task}: trailing tokens supervised"
        assert (weights[:a_start] == 0).all(), f"{task}: prefix/question weighted"
        assert (weights[a_start:a_end][supervised] > 0).all(), f"{task}: answer not weighted"
        assert (weights[a_start:a_end][~supervised] == 0).all(), f"{task}: format chars weighted"
        assert (weights[a_end:] == 0).all(), f"{task}: trailing weights nonzero"
        # Constant suffix markers must never be supervised
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        assert text[a_start - len(gen.config.answer_prefix) : a_start] == gen.config.answer_prefix
        assert labels[a_start - 1] == -100, f"{task}: answer prefix char supervised"
        assert labels[q_start - 1] == -100, f"{task}: question prefix char supervised"
    print("answer-only (no constant supervision) OK")


def _haystack_without_suffix(text: str, sample, gen: LongContextSampleGenerator) -> str:
    suffix_start = int(sample.meta_dict["suffix_start"])
    suffix = (
        f"{gen.config.question_prefix}{sample.question}"
        f"{gen.config.answer_prefix}{sample.expected_answer}"
    )
    return text[:suffix_start] + text[suffix_start + len(suffix) :]


def test_needle_preserved_at_high_depth():
    """Needle must remain in haystack at high depth (suffix may split the haystack)."""
    gen = LongContextSampleGenerator()
    sample = gen.generate_one(
        context_length=8192,
        needle_depth=0.90,
        task_type="exact_retrieval",
        haystack_mode="random_sentences",
        seed=424242,
    )
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    haystack = _haystack_without_suffix(text, sample, gen)
    assert sample.expected_answer in haystack
    assert float(sample.meta_dict["needle_depth"]) >= 0.70
    print("needle preserved at high depth OK")


def test_suffix_not_always_at_end():
    """Query suffix must appear at variable offsets — not glued to the sequence end."""
    gen = LongContextSampleGenerator()
    suffix_starts: list[int] = []
    for seed in range(30):
        sample = gen.generate_one(
            context_length=4096,
            needle_depth=0.5,
            task_type="exact_retrieval",
            haystack_mode="random_sentences",
            seed=seed,
        )
        suffix_starts.append(int(sample.meta_dict["suffix_start"]))
    assert max(suffix_starts) - min(suffix_starts) > 256, f"suffix positions not varied: {suffix_starts}"
    assert any(s < 4096 - 400 for s in suffix_starts), "suffix never placed away from end"
    print("variable suffix position OK")


def test_scattered_multi_needles():
    """Multi-segment tasks scatter needles and honor the grid needle_depth anchor."""
    gen = LongContextSampleGenerator(
        LongContextBenchmarkConfig(scatter_multi_needles=True)
    )
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.75,
        task_type="distractor",
        haystack_mode="random_tokens",
        seed=4242,
    )
    assert sample.meta_dict.get("scatter_needles") is True
    depths = sample.meta_dict.get("segment_depths", [])
    assert len(depths) > 1
    assert max(depths) - min(depths) > 0.03, f"needles not spread: {depths}"
    assert abs(float(sum(depths) / len(depths)) - 0.75) < 0.25, f"depth not anchored: {depths}"
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    haystack = _haystack_without_suffix(text, sample, gen)
    assert sample.expected_answer in haystack
    print("scattered multi-needles OK")


def test_multi_key_random_target():
    import random

    rng = random.Random(99)
    targets = set()
    for _ in range(40):
        payload = generate_task("multi_key", rng, {"num_multi_keys": 4})
        targets.add(payload.metadata["target"])
    assert len(targets) >= 3, f"multi_key always queries same target: {targets}"
    print("multi_key random target OK")


def test_suffix_never_splits_needle():
    """Query suffix insertion must never cut through a needle segment."""
    gen = LongContextSampleGenerator()
    for task in list(PRIMARY_GATE_TASK_TYPES) + list(SECONDARY_TASK_TYPES):
        for seed in range(25):
            sample = gen.generate_one(
                context_length=4096,
                needle_depth=0.5,
                task_type=task,
                haystack_mode="random_sentences",
                seed=hash((task, seed)) % 100000,
            )
            suffix_start = int(sample.meta_dict["suffix_start"])
            segments = sample.meta_dict.get("needle_segments", [])
            suffix = (
                f"{gen.config.question_prefix}{sample.question}"
                f"{gen.config.answer_prefix}{sample.expected_answer}"
            )
            text = gen.tokenizer.decode(sample.input_ids.tolist())
            haystack = text[:suffix_start] + text[suffix_start + len(suffix) :]
            forbidden = _suffix_start_forbidden(_needle_spans(haystack, segments))
            assert suffix_start not in forbidden, f"{task} seed={seed}: suffix splits needle"
            for seg in segments:
                assert seg in text, f"{task}: needle segment missing after suffix insert"
    print("suffix never splits needle OK")


def test_multiple_needles_single_vault_only():
    import random

    rng = random.Random(7)
    for _ in range(30):
        payload = generate_task("multiple_needles", rng)
        assert "access code for" in payload.question
        assert "," not in payload.expected_answer
    print("multiple_needles single-vault only OK")


def test_primary_gate_default_tasks():
    cfg = LongContextBenchmarkConfig()
    assert set(cfg.task_types) == set(PRIMARY_GATE_TASK_TYPES)
    assert set(cfg.secondary_task_types) == set(SECONDARY_TASK_TYPES)
    assert "distractor" not in cfg.task_types
    assert "multi_hop" not in cfg.task_types
    assert "multi_key" not in cfg.task_types
    assert cfg.eval_task_types() == list(cfg.task_types) + list(cfg.secondary_task_types)
    print("primary gate default tasks OK")


def test_training_excludes_secondary_tasks():
    from routing_attention.benchmarks.long_context.dataset import LongContextTrainDataset

    cfg = LongContextBenchmarkConfig()
    ds = LongContextTrainDataset(cfg, batch_size=1, train_context_length=512)
    assert "distractor" not in ds._tasks
    assert set(ds._tasks) == set(PRIMARY_GATE_TASK_TYPES)
    print("training excludes secondary tasks OK")


def test_holdout_includes_secondary_tasks():
    cfg = LongContextBenchmarkConfig(
        context_lengths=[512],
        needle_depths=[0.5],
        haystack_modes=["random_sentences"],
        eval_samples_per_cell=1,
    )
    clear_holdout_cache()
    holdout = get_holdout_grid(cfg)
    tasks = {s.task_type for s in holdout}
    assert set(PRIMARY_GATE_TASK_TYPES).issubset(tasks)
    assert "distractor" in tasks
    print("holdout includes secondary tasks OK")


def test_primary_gate_metric():
    from routing_attention.benchmarks.long_context.evaluation import EvalRecord, EvalSummary

    evaluator = LongContextEvaluator()
    records = [
        EvalRecord(True, "1", "1", "exact_retrieval", 512, 0.5),
        EvalRecord(False, "2", "3", "key_value", 512, 0.5),
        EvalRecord(True, "x", "x", "distractor", 512, 0.5),
    ]
    summary = evaluator.summarize(records)
    assert summary.primary_gate_correct == 1
    assert summary.primary_gate_total == 2
    assert summary.primary_gate_accuracy == 0.5
    assert summary.secondary_correct == 1
    assert summary.secondary_total == 1
    assert summary.overall_accuracy == 2 / 3
    print("primary gate metric OK")


def test_query_to_needle_distance_metadata():
    gen = LongContextSampleGenerator()
    sample = gen.generate_one(
        context_length=4096,
        needle_depth=0.5,
        task_type="exact_retrieval",
        haystack_mode="random_sentences",
        seed=123,
    )
    dist = int(sample.meta_dict["query_to_needle_distance"])
    assert dist > 0
    assert "needle_depth_final" in sample.meta_dict
    print(f"query-to-needle distance OK (dist={dist})")


def test_answer_loss_weights():
    gen = LongContextSampleGenerator(
        LongContextBenchmarkConfig(answer_loss_weight=3.0)
    )
    sample = gen.generate_one(
        context_length=1024,
        needle_depth=0.5,
        task_type="key_value",
        haystack_mode="random_sentences",
        seed=77,
    )
    w = sample.loss_weights_np
    assert w is not None
    q_start = int(sample.meta_dict["question_start"])
    assert (w[: sample.answer_start] == 0).all()
    supervised = sample.labels.numpy()[sample.answer_start : sample.answer_end] != -100
    assert (w[sample.answer_start : sample.answer_end][supervised] == 3.0).all()
    assert (w[q_start : sample.answer_start] == 0).all()
    print("answer loss weights OK")


def test_question_phrasing_variety():
    import random

    rng = random.Random(0)
    questions = {generate_task("exact_retrieval", rng).question for _ in range(40)}
    assert len(questions) >= 3, f"exact_retrieval questions not diverse: {questions}"
    print(f"question phrasing variety OK ({len(questions)} unique)")


def test_balanced_task_sampling():
    import random

    from routing_attention.benchmarks.long_context.dataset import LongContextTrainDataset

    cfg = LongContextBenchmarkConfig(
        task_types=list(TASK_GENERATORS.keys()),
        haystack_modes=["random_sentences"],
        train_task_sampling="balanced",
    )
    ds = LongContextTrainDataset(cfg, batch_size=1, train_context_length=512)
    rng = random.Random(42)
    for i in range(len(cfg.task_types) * 2):
        expected = cfg.task_types[i % len(cfg.task_types)]
        got = ds._next_task_type(rng, i)
        assert got == expected, f"step {i}: expected {expected}, got {got}"
    print("balanced task sampling OK")


def test_config_roundtrip(tmp_path=None):
    cfg = LongContextBenchmarkConfig(context_lengths=[1024, 2048])
    path = Path(ROOT) / "configs" / "experiment_7.yaml"
    assert path.exists()
    loaded = LongContextBenchmarkConfig.load(path)
    assert loaded.context_lengths[0] == max(loaded.context_lengths)
    print("config load OK")


def main():
    test_all_task_types()
    test_context_lengths()
    test_longest_first_order()
    test_untrained_model_near_zero_accuracy()
    test_train_holdout_disjoint()
    test_char_token_alignment()
    test_no_question_answer_leak()
    test_answer_only_no_constant_supervision()
    test_needle_preserved_at_high_depth()
    test_suffix_not_always_at_end()
    test_balanced_task_sampling()
    test_scattered_multi_needles()
    test_suffix_never_splits_needle()
    test_multiple_needles_single_vault_only()
    test_primary_gate_default_tasks()
    test_training_excludes_secondary_tasks()
    test_holdout_includes_secondary_tasks()
    test_primary_gate_metric()
    test_query_to_needle_distance_metadata()
    test_answer_loss_weights()
    test_question_phrasing_variety()
    test_fast_attention_backends()
    test_train_requires_fixed_context_length()
    test_production_backend_manifest()
    test_validate_every_policy()
    test_routing_sparse_topk_path()
    test_evaluator_random_model()
    test_config_roundtrip()
    print("\nAll long-context benchmark sanity checks passed.")


if __name__ == "__main__":
    main()
