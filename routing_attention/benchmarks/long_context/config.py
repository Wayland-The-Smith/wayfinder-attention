"""Configuration for the procedural long-context retrieval benchmark (Experiment 7)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Longest-first: eval/training sweeps surface OOM and shape errors immediately.
DEFAULT_CONTEXT_LENGTHS = [32768, 16384, 8192, 4096, 2048, 1024]
DRY_RUN_CONTEXT_LENGTHS = [8192, 2048, 512]
DEFAULT_NEEDLE_DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]
DEFAULT_TASK_TYPES = [
    "exact_retrieval",
    "key_value",
    "multiple_needles",
]
DEFAULT_SECONDARY_TASK_TYPES = ["distractor"]
DEFAULT_HAYSTACK_MODES = ["random_tokens", "random_sentences", "structured_records"]
DEFAULT_HOLDOUT_SEED = 1_000_042
# Bump when task/grid semantics change (invalidates holdout cache).
BENCHMARK_VERSION = 7
SYNTHETIC_BENCHMARK_VERSION = 21
BENCHMARK_FAMILIES = ("nl", "synthetic")
# Training supervises variable answer tokens only (prefixes/questions masked).
TRAIN_LABEL_MODES = ("answer_only", "query_only_answer")
SLOT_QUAD_PLACEMENTS = ("random", "fixed_grid")
MODEL_OUTPUT_HEADS = ("lm_token", "pointer_index", "pointer_mlp", "pool_mlp_token")
POINTER_TARGET_MODES = ("value_slots", "full_sequence")


def apply_synthetic_family_profile(cfg: LongContextBenchmarkConfig) -> LongContextBenchmarkConfig:
    """Resolve synthetic NIAH vs token-native slot_pointer profile from task_types."""
    if cfg.benchmark_family != "synthetic":
        return cfg
    tasks = list(cfg.task_types or [])
    if tasks == ["slot_pointer"]:
        return cfg.apply_slot_pointer_profile()
    if tasks == ["mqar_addr_val"]:
        return cfg.apply_mqar_addr_val_profile()
    return cfg.apply_synthetic_profile()
TRAIN_TASK_SAMPLINGS = ("uniform", "balanced", "retrieval_heavy")
SUFFIX_PLACEMENTS = ("random", "after_needles", "at_end", "causal_safe_random")


@dataclass
class LongContextBenchmarkConfig:
    """Serializable benchmark configuration."""

    context_lengths: list[int] = field(
        default_factory=lambda: sorted(set(DEFAULT_CONTEXT_LENGTHS), reverse=True)
    )
    needle_depths: list[float] = field(default_factory=lambda: list(DEFAULT_NEEDLE_DEPTHS))
    task_types: list[str] = field(default_factory=lambda: list(DEFAULT_TASK_TYPES))
    # Eval-only tasks (excluded from training and primary gate).
    secondary_task_types: list[str] = field(
        default_factory=lambda: list(DEFAULT_SECONDARY_TASK_TYPES)
    )
    haystack_modes: list[str] = field(default_factory=lambda: list(DEFAULT_HAYSTACK_MODES))
    vocab_size: int = 256
    # Training stream base seed (infinite random procedural samples).
    seed: int = 45
    # Fixed held-out eval grid — never used during training.
    holdout_seed: int = DEFAULT_HOLDOUT_SEED
    question_prefix: str = " Question: "
    answer_prefix: str = " Answer: "
    num_distractors: int = 8
    num_multi_keys: int = 3
    num_needles_multi: int = 3
    tinystories_path: str | None = None
    eval_samples_per_cell: int = 16
    max_answer_chars: int = 64
    shared_init: bool = True
    eval_grid_workers: int = 0
    benchmark_version: int = BENCHMARK_VERSION
    # ``nl``: natural-language NIAH (Experiment 7 default). ``synthetic``: L0–L4 pointer suite.
    benchmark_family: str = "nl"
    synthetic_hop_count: int = 2
    synthetic_hop_count_min: int = 2
    synthetic_hop_count_max: int = 4
    synthetic_decoy_keys: int = 5
    synthetic_decoy_addrs: int = 4
    synthetic_fake_ptrs: int = 2
    # ``massive_addr_val`` / ``mqar_addr_val``: number of independent ADDR→VAL bindings.
    num_kv_pairs: int = 50
    # ``mqar_addr_val``: number of QUERY tokens in the suffix (supervise last query's value).
    num_queries: int = 1
    # When true, supervise all query answers (space-separated) instead of last query only.
    mqar_supervise_all_queries: bool = False
    # Char-level answer width (1–6 digits). Passkey tasks use 4–6.
    answer_digit_width: int = 2
    # addr_val_conflict*: same address, different values — last / first / middle wins.
    synthetic_conflict_rows: int = 3
    # slot_pointer: number of contiguous ``[addr][;][value][,]`` quads per sample.
    num_slot_quads: int = 50
    # ``random`` | ``fixed_grid`` — quad start positions in slot_pointer content.
    slot_quad_placement: str = "random"
    # Require semantic tokens only inside quads (+ query addr at T-1); hay uses 103..128 only.
    slot_enforce_unique_semantics: bool = False
    # When false, suffix is question-only (answer not present in model input).
    include_answer_in_suffix: bool = True
    # Optional label for variant-specific holdout offsets / logging.
    benchmark_variant: str = ""
    # Supervise answer span only — never prefixes, questions, or haystack.
    train_label_mode: str = "answer_only"
    # ``balanced``: round-robin tasks; ``retrieval_heavy``: oversample retrieval tasks.
    train_task_sampling: str = "retrieval_heavy"
    # Insert multi-segment needles at independent depths (true scattered NIAH).
    scatter_multi_needles: bool = True
    # Random needle placement bounds (chars) for synthetic hop-first protocol.
    scatter_placement_min: int = 0
    scatter_placement_max: int | None = None
    # Optional placement warmup: [{"until_step": N, "scatter_placement_min": lo, "scatter_placement_max": hi}]
    needle_scatter_curriculum: list[dict] = field(default_factory=list)
    # Optional ptr_chain difficulty ramp:
    # [{"until_step": N, "synthetic_hop_count": H, "num_distractors": D, ...}, ...]
    synthetic_task_curriculum: list[dict] = field(default_factory=list)
    # CE weight on supervised answer payload tokens (sparse signal at long T).
    answer_loss_weight: float = 8.0
    # Query suffix placement — see SUFFIX_PLACEMENTS. ``at_end`` = classic NIAH (query at sequence end).
    suffix_depth_min: float = 0.10
    suffix_depth_max: float = 0.90
    suffix_placement: str = "random"
    suffix_after_needles_gap_max: int = 32
    min_haystack_side_chars: int = 128
    generation_max_attempts: int = 16
    # Optional length curriculum: [{"until_step": N, "context_length": T}, ...]
    context_curriculum: list[dict] = field(default_factory=list)
    # Optional suffix curriculum: [{"until_step": N, "suffix_placement": ..., ...}, ...]
    suffix_curriculum: list[dict] = field(default_factory=list)
    overfit_train_samples: int = 0
    # Training only: repeat the same needles + query for N consecutive batches (placement varies).
    placement_episode_batches: int = 0

    def __post_init__(self) -> None:
        if self.train_label_mode not in TRAIN_LABEL_MODES:
            raise ValueError(f"train_label_mode must be one of {TRAIN_LABEL_MODES}")
        if self.train_task_sampling not in TRAIN_TASK_SAMPLINGS:
            raise ValueError(f"train_task_sampling must be one of {TRAIN_TASK_SAMPLINGS}")
        if self.benchmark_family not in BENCHMARK_FAMILIES:
            raise ValueError(f"benchmark_family must be one of {BENCHMARK_FAMILIES}")
        if self.suffix_placement not in SUFFIX_PLACEMENTS:
            raise ValueError(f"suffix_placement must be one of {SUFFIX_PLACEMENTS}")
        if self.overfit_train_samples < 0:
            raise ValueError("overfit_train_samples must be >= 0")
        if self.num_kv_pairs < 2:
            raise ValueError("num_kv_pairs must be >= 2")
        if self.num_queries < 1:
            raise ValueError("num_queries must be >= 1")
        if self.answer_digit_width < 1 or self.answer_digit_width > 6:
            raise ValueError("answer_digit_width must be in [1, 6]")
        if self.num_slot_quads < 1 or self.num_slot_quads > 100:
            raise ValueError("num_slot_quads must be in [1, 100]")
        if self.slot_quad_placement not in SLOT_QUAD_PLACEMENTS:
            raise ValueError(f"slot_quad_placement must be one of {SLOT_QUAD_PLACEMENTS}")
        if self.placement_episode_batches < 0:
            raise ValueError("placement_episode_batches must be >= 0")
        self.context_lengths = sorted(set(self.context_lengths), reverse=True)

    def primary_gate_task_types(self) -> tuple[str, ...]:
        if self.benchmark_family == "synthetic":
            from routing_attention.benchmarks.long_context.tasks_synthetic import (
                SYNTHETIC_PRIMARY_GATE_TASK_TYPES,
            )

            return SYNTHETIC_PRIMARY_GATE_TASK_TYPES
        from routing_attention.benchmarks.long_context.tasks import PRIMARY_GATE_TASK_TYPES

        return PRIMARY_GATE_TASK_TYPES

    def secondary_eval_task_types(self) -> tuple[str, ...]:
        if self.benchmark_family == "synthetic":
            from routing_attention.benchmarks.long_context.tasks_synthetic import (
                SYNTHETIC_SECONDARY_TASK_TYPES,
            )

            return SYNTHETIC_SECONDARY_TASK_TYPES
        from routing_attention.benchmarks.long_context.tasks import SECONDARY_TASK_TYPES

        return SECONDARY_TASK_TYPES

    def eval_task_types(self) -> list[str]:
        """All task types evaluated on holdout (same set as training for synthetic)."""
        return list(dict.fromkeys(list(self.task_types) + list(self.secondary_task_types)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LongContextBenchmarkConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        # Legacy field removed — training is step-based, not epoch-based.
        filtered.pop("train_samples_per_epoch", None)
        cfg = cls(**filtered)
        return cfg.normalized()

    def normalized(self) -> LongContextBenchmarkConfig:
        lengths = sorted(set(self.context_lengths), reverse=True)
        return LongContextBenchmarkConfig(
            context_lengths=lengths,
            needle_depths=list(self.needle_depths),
            task_types=list(self.task_types),
            secondary_task_types=list(self.secondary_task_types),
            haystack_modes=list(self.haystack_modes),
            vocab_size=self.vocab_size,
            seed=self.seed,
            holdout_seed=self.holdout_seed,
            question_prefix=self.question_prefix,
            answer_prefix=self.answer_prefix,
            num_distractors=self.num_distractors,
            num_multi_keys=self.num_multi_keys,
            num_needles_multi=self.num_needles_multi,
            tinystories_path=self.tinystories_path,
            eval_samples_per_cell=self.eval_samples_per_cell,
            max_answer_chars=self.max_answer_chars,
            shared_init=self.shared_init,
            eval_grid_workers=self.eval_grid_workers,
            benchmark_version=self.benchmark_version,
            benchmark_family=self.benchmark_family,
            synthetic_hop_count=self.synthetic_hop_count,
            synthetic_hop_count_min=self.synthetic_hop_count_min,
            synthetic_hop_count_max=self.synthetic_hop_count_max,
            synthetic_decoy_keys=self.synthetic_decoy_keys,
            synthetic_decoy_addrs=self.synthetic_decoy_addrs,
            synthetic_fake_ptrs=self.synthetic_fake_ptrs,
            num_kv_pairs=self.num_kv_pairs,
            num_queries=self.num_queries,
            mqar_supervise_all_queries=self.mqar_supervise_all_queries,
            answer_digit_width=self.answer_digit_width,
            synthetic_conflict_rows=self.synthetic_conflict_rows,
            num_slot_quads=self.num_slot_quads,
            slot_quad_placement=self.slot_quad_placement,
            slot_enforce_unique_semantics=self.slot_enforce_unique_semantics,
            include_answer_in_suffix=self.include_answer_in_suffix,
            benchmark_variant=self.benchmark_variant,
            train_label_mode=self.train_label_mode,
            train_task_sampling=self.train_task_sampling,
            scatter_multi_needles=self.scatter_multi_needles,
            scatter_placement_min=self.scatter_placement_min,
            scatter_placement_max=self.scatter_placement_max,
            needle_scatter_curriculum=list(self.needle_scatter_curriculum),
            synthetic_task_curriculum=list(self.synthetic_task_curriculum),
            answer_loss_weight=self.answer_loss_weight,
            suffix_depth_min=self.suffix_depth_min,
            suffix_depth_max=self.suffix_depth_max,
            suffix_placement=self.suffix_placement,
            suffix_after_needles_gap_max=self.suffix_after_needles_gap_max,
            min_haystack_side_chars=self.min_haystack_side_chars,
            generation_max_attempts=self.generation_max_attempts,
            context_curriculum=list(self.context_curriculum),
            suffix_curriculum=list(self.suffix_curriculum),
            overfit_train_samples=self.overfit_train_samples,
            placement_episode_batches=self.placement_episode_batches,
        )

    def holdout_config(self) -> LongContextBenchmarkConfig:
        """Config for the fixed held-out eval grid (shared across all variants)."""
        cfg = self.normalized()
        cfg.seed = self.holdout_seed
        cfg.placement_episode_batches = 0
        return cfg

    def cache_key(self) -> tuple:
        """Hashable key for holdout grid caching."""

        def _freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
            if isinstance(value, list):
                return tuple(_freeze(v) for v in value)
            return value

        d = self.holdout_config().to_dict()
        return tuple(sorted((k, _freeze(v)) for k, v in d.items()))

    def apply_dry_run_profile(self) -> LongContextBenchmarkConfig:
        return LongContextBenchmarkConfig(
            context_lengths=list(DRY_RUN_CONTEXT_LENGTHS),
            needle_depths=[0.10, 0.50, 0.90],
            task_types=list(DEFAULT_TASK_TYPES),
            secondary_task_types=list(self.secondary_task_types),
            haystack_modes=["random_sentences"],
            vocab_size=self.vocab_size,
            seed=self.seed,
            holdout_seed=self.holdout_seed + 9_999,
            question_prefix=self.question_prefix,
            answer_prefix=self.answer_prefix,
            num_distractors=self.num_distractors,
            num_multi_keys=self.num_multi_keys,
            num_needles_multi=self.num_needles_multi,
            tinystories_path=self.tinystories_path,
            eval_samples_per_cell=1,
            max_answer_chars=self.max_answer_chars,
            shared_init=self.shared_init,
            eval_grid_workers=0,
            benchmark_version=self.benchmark_version,
            benchmark_family=self.benchmark_family,
            synthetic_hop_count=self.synthetic_hop_count,
            synthetic_hop_count_min=self.synthetic_hop_count_min,
            synthetic_hop_count_max=self.synthetic_hop_count_max,
            synthetic_decoy_keys=self.synthetic_decoy_keys,
            synthetic_decoy_addrs=self.synthetic_decoy_addrs,
            synthetic_fake_ptrs=self.synthetic_fake_ptrs,
            num_kv_pairs=self.num_kv_pairs,
            num_queries=self.num_queries,
            mqar_supervise_all_queries=self.mqar_supervise_all_queries,
            answer_digit_width=self.answer_digit_width,
            synthetic_conflict_rows=self.synthetic_conflict_rows,
            num_slot_quads=self.num_slot_quads,
            train_label_mode=self.train_label_mode,
            train_task_sampling=self.train_task_sampling,
            scatter_multi_needles=self.scatter_multi_needles,
            scatter_placement_min=self.scatter_placement_min,
            scatter_placement_max=self.scatter_placement_max,
            needle_scatter_curriculum=list(self.needle_scatter_curriculum),
            answer_loss_weight=self.answer_loss_weight,
            suffix_depth_min=self.suffix_depth_min,
            suffix_depth_max=self.suffix_depth_max,
            suffix_placement=self.suffix_placement,
            suffix_after_needles_gap_max=self.suffix_after_needles_gap_max,
            min_haystack_side_chars=self.min_haystack_side_chars,
            generation_max_attempts=self.generation_max_attempts,
            context_curriculum=list(self.context_curriculum),
            suffix_curriculum=list(self.suffix_curriculum),
            overfit_train_samples=self.overfit_train_samples,
            placement_episode_batches=self.placement_episode_batches,
        ).normalized()

    def apply_synthetic_profile(self) -> LongContextBenchmarkConfig:
        """L0–L4 synthetic pointer / address NIAH suite (minimal vocabulary)."""
        from routing_attention.benchmarks.long_context.tasks_synthetic import SYNTHETIC_TASK_TYPES

        requested = list(self.task_types)
        if requested and all(t in SYNTHETIC_TASK_TYPES for t in requested):
            task_types = requested
        else:
            task_types = list(SYNTHETIC_TASK_TYPES)

        return LongContextBenchmarkConfig(
            context_lengths=list(self.context_lengths) or [8192, 4096, 2048, 1024],
            needle_depths=list(self.needle_depths),
            task_types=task_types,
            secondary_task_types=[],
            haystack_modes=["synthetic_noise"],
            vocab_size=128,
            seed=self.seed,
            holdout_seed=self.holdout_seed,
            question_prefix="",
            answer_prefix=" A ",
            num_distractors=self.num_distractors,
            num_multi_keys=self.num_multi_keys,
            num_needles_multi=self.num_needles_multi,
            tinystories_path=self.tinystories_path,
            eval_samples_per_cell=self.eval_samples_per_cell,
            max_answer_chars=self.max_answer_chars,
            shared_init=self.shared_init,
            eval_grid_workers=self.eval_grid_workers,
            benchmark_version=SYNTHETIC_BENCHMARK_VERSION,
            benchmark_family="synthetic",
            synthetic_hop_count=self.synthetic_hop_count,
            synthetic_hop_count_min=self.synthetic_hop_count,
            synthetic_hop_count_max=self.synthetic_hop_count,
            synthetic_decoy_keys=self.synthetic_decoy_keys,
            synthetic_decoy_addrs=self.synthetic_decoy_addrs,
            synthetic_fake_ptrs=0,
            num_kv_pairs=self.num_kv_pairs,
            num_queries=self.num_queries,
            answer_digit_width=self.answer_digit_width,
            synthetic_conflict_rows=self.synthetic_conflict_rows,
            num_slot_quads=self.num_slot_quads,
            slot_quad_placement=self.slot_quad_placement,
            slot_enforce_unique_semantics=self.slot_enforce_unique_semantics,
            include_answer_in_suffix=self.include_answer_in_suffix,
            benchmark_variant=self.benchmark_variant,
            train_label_mode=self.train_label_mode,
            train_task_sampling=self.train_task_sampling,
            scatter_multi_needles=self.scatter_multi_needles,
            scatter_placement_min=self.scatter_placement_min,
            scatter_placement_max=self.scatter_placement_max,
            needle_scatter_curriculum=list(self.needle_scatter_curriculum),
            synthetic_task_curriculum=list(self.synthetic_task_curriculum),
            answer_loss_weight=self.answer_loss_weight,
            suffix_depth_min=self.suffix_depth_min,
            suffix_depth_max=self.suffix_depth_max,
            suffix_placement="at_end",
            suffix_after_needles_gap_max=self.suffix_after_needles_gap_max,
            min_haystack_side_chars=self.min_haystack_side_chars,
            generation_max_attempts=self.generation_max_attempts,
            context_curriculum=list(self.context_curriculum),
            suffix_curriculum=list(self.suffix_curriculum),
            overfit_train_samples=self.overfit_train_samples,
            placement_episode_batches=self.placement_episode_batches,
        ).normalized()

    def apply_mqar_addr_val_profile(self) -> LongContextBenchmarkConfig:
        """MQAR many-binding addr_val @ fixed T — query-only or all-query answer supervision."""
        base = self.apply_synthetic_profile()
        depths = list(self.needle_depths) if self.needle_depths else [0.5]
        supervise_all = bool(self.mqar_supervise_all_queries)
        return LongContextBenchmarkConfig(
            context_lengths=list(base.context_lengths) or [512],
            needle_depths=depths,
            task_types=["mqar_addr_val"],
            secondary_task_types=[],
            haystack_modes=["synthetic_noise"],
            vocab_size=128,
            seed=self.seed,
            holdout_seed=self.holdout_seed,
            question_prefix="",
            answer_prefix="",
            num_distractors=0,
            num_kv_pairs=self.num_kv_pairs,
            num_queries=self.num_queries,
            mqar_supervise_all_queries=supervise_all,
            answer_digit_width=max(1, self.answer_digit_width),
            eval_samples_per_cell=self.eval_samples_per_cell,
            max_answer_chars=self.max_answer_chars,
            shared_init=self.shared_init,
            eval_grid_workers=self.eval_grid_workers,
            benchmark_version=SYNTHETIC_BENCHMARK_VERSION,
            benchmark_family="synthetic",
            benchmark_variant=self.benchmark_variant or "mqar_addr_val_calibration",
            include_answer_in_suffix=supervise_all,
            train_label_mode="answer_only" if supervise_all else "query_only_answer",
            train_task_sampling=self.train_task_sampling,
            scatter_multi_needles=self.scatter_multi_needles,
            scatter_placement_min=self.scatter_placement_min,
            scatter_placement_max=self.scatter_placement_max,
            answer_loss_weight=self.answer_loss_weight,
            suffix_placement="at_end",
            min_haystack_side_chars=self.min_haystack_side_chars,
            generation_max_attempts=self.generation_max_attempts,
            overfit_train_samples=self.overfit_train_samples,
            placement_episode_batches=self.placement_episode_batches,
        ).normalized()

    def apply_slot_pointer_profile(self) -> LongContextBenchmarkConfig:
        """Token-native slot-pointer task @ fixed T with 129-token vocabulary."""
        return LongContextBenchmarkConfig(
            context_lengths=list(self.context_lengths) or [2048],
            needle_depths=list(self.needle_depths) or [0.5],
            task_types=["slot_pointer"],
            secondary_task_types=[],
            haystack_modes=["synthetic_noise"],
            vocab_size=129,
            seed=self.seed,
            holdout_seed=self.holdout_seed,
            question_prefix="",
            answer_prefix="",
            num_distractors=0,
            eval_samples_per_cell=self.eval_samples_per_cell,
            max_answer_chars=self.max_answer_chars,
            shared_init=self.shared_init,
            eval_grid_workers=self.eval_grid_workers,
            benchmark_version=SYNTHETIC_BENCHMARK_VERSION,
            benchmark_family="synthetic",
            num_slot_quads=self.num_slot_quads,
            slot_quad_placement=self.slot_quad_placement,
            slot_enforce_unique_semantics=self.slot_enforce_unique_semantics,
            include_answer_in_suffix=self.include_answer_in_suffix,
            benchmark_variant=self.benchmark_variant,
            train_label_mode=self.train_label_mode,
            train_task_sampling=self.train_task_sampling,
            scatter_multi_needles=False,
            suffix_placement="at_end",
            min_haystack_side_chars=self.min_haystack_side_chars,
            generation_max_attempts=self.generation_max_attempts,
        ).normalized()

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> LongContextBenchmarkConfig:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            return cls.from_dict(yaml.safe_load(text) or {})
        return cls.from_dict(json.loads(text))
