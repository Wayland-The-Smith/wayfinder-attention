"""Procedural long-context retrieval benchmark (Experiment 7)."""

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    LongContextEvalDataset,
    LongContextTrainDataset,
    get_long_context_dataloader,
)
from routing_attention.benchmarks.long_context.evaluation import EvalSummary, LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import LongContextSample, LongContextSampleGenerator
from routing_attention.benchmarks.long_context.comparison import save_comparison_plots
from routing_attention.benchmarks.long_context.production_backends import (
    EXP7_PRODUCTION_BACKENDS,
    production_manifest_for_variants,
)
from routing_attention.benchmarks.long_context.holdout import (
    apply_holdout_total_samples,
    clear_holdout_cache,
    count_holdout_grid_cells,
    filter_holdout_by_context_length,
    get_holdout_grid,
    resolve_eval_samples_per_cell,
    resolve_holdout_splits,
)
from routing_attention.benchmarks.long_context.plots import save_all_benchmark_plots
from routing_attention.benchmarks.long_context.report import generate_markdown_report
from routing_attention.benchmarks.long_context.success_criteria import evaluate_success_criteria

__all__ = [
    "LongContextBenchmarkConfig",
    "LongContextEvalDataset",
    "LongContextEvaluator",
    "LongContextSample",
    "LongContextSampleGenerator",
    "LongContextTrainDataset",
    "EvalSummary",
    "get_long_context_dataloader",
    "save_all_benchmark_plots",
    "save_comparison_plots",
    "generate_markdown_report",
    "evaluate_success_criteria",
    "get_holdout_grid",
    "clear_holdout_cache",
    "filter_holdout_by_context_length",
    "resolve_holdout_splits",
    "resolve_eval_samples_per_cell",
    "apply_holdout_total_samples",
    "count_holdout_grid_cells",
    "EXP7_PRODUCTION_BACKENDS",
    "production_manifest_for_variants",
]
