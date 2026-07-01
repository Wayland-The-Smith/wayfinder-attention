from routing_attention.evaluation.recall import (
    compute_recall_at_k,
    compute_recall_by_distance,
    compute_random_routing_baseline,
    compute_recall_by_distance_from_router,
    compute_recall_from_router,
    evaluate_per_layer_recall,
    prepare_eval_tensors,
    resolve_max_eval_tokens,
    subsample_for_eval,
    subsample_tokens_for_eval,
)
from routing_attention.evaluation.metrics import evaluate_lm, compute_perplexity
from routing_attention.evaluation.benchmarking import benchmark_attention, measure_memory_usage

try:
    from routing_attention.evaluation.plots import (
        plot_recall_by_distance,
        plot_comparison_bar,
        save_all_plots,
    )
except ImportError:
    plot_recall_by_distance = None  # type: ignore[misc, assignment]
    plot_comparison_bar = None  # type: ignore[misc, assignment]
    save_all_plots = None  # type: ignore[misc, assignment]

__all__ = [
    "compute_recall_at_k",
    "compute_recall_by_distance",
    "compute_random_routing_baseline",
    "compute_recall_by_distance_from_router",
    "compute_recall_from_router",
    "evaluate_per_layer_recall",
    "prepare_eval_tensors",
    "resolve_max_eval_tokens",
    "subsample_for_eval",
    "subsample_tokens_for_eval",
    "evaluate_lm",
    "compute_perplexity",
    "benchmark_attention",
    "measure_memory_usage",
    "plot_recall_by_distance",
    "plot_comparison_bar",
    "save_all_plots",
]
