"""Evaluation for procedural long-context retrieval benchmark."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dataset import (
    LongContextEvalDataset,
    transfer_batch_to_device,
)
from routing_attention.benchmarks.long_context.generator import LongContextSample, LongContextSampleGenerator
from routing_attention.benchmarks.long_context.holdout import get_holdout_grid
from routing_attention.benchmarks.long_context.tokenizer import BenchmarkTokenizer


@dataclass
class EvalRecord:
    correct: bool
    predicted: str
    expected: str
    task_type: str
    context_length: int
    needle_depth: float
    haystack_mode: str = ""


@dataclass
class EvalError:
    context_length: int
    task_type: str
    needle_depth: float
    error: str


@dataclass
class EvalSummary:
    overall_accuracy: float
    total: int
    correct: int
    primary_gate_accuracy: float = 0.0
    primary_gate_total: int = 0
    primary_gate_correct: int = 0
    secondary_accuracy: float = 0.0
    secondary_total: int = 0
    secondary_correct: int = 0
    # Legacy alias for primary_gate_*.
    pure_niah_accuracy: float = 0.0
    pure_niah_total: int = 0
    pure_niah_correct: int = 0
    by_context_length: dict[int, float] = field(default_factory=dict)
    by_needle_depth: dict[str, float] = field(default_factory=dict)
    by_task_type: dict[str, float] = field(default_factory=dict)
    by_cell: dict[str, float] = field(default_factory=dict)
    records: list[EvalRecord] = field(default_factory=list)
    errors: list[EvalError] = field(default_factory=list)
    skipped_after_error: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "total": self.total,
            "correct": self.correct,
            "primary_gate_accuracy": self.primary_gate_accuracy,
            "primary_gate_total": self.primary_gate_total,
            "primary_gate_correct": self.primary_gate_correct,
            "secondary_accuracy": self.secondary_accuracy,
            "secondary_total": self.secondary_total,
            "secondary_correct": self.secondary_correct,
            "pure_niah_accuracy": self.primary_gate_accuracy,
            "pure_niah_total": self.primary_gate_total,
            "pure_niah_correct": self.primary_gate_correct,
            "by_context_length": self.by_context_length,
            "by_needle_depth": self.by_needle_depth,
            "by_task_type": self.by_task_type,
            "by_cell": self.by_cell,
            "errors": [
                {
                    "context_length": e.context_length,
                    "task_type": e.task_type,
                    "needle_depth": e.needle_depth,
                    "error": e.error,
                }
                for e in self.errors
            ],
            "skipped_after_error": self.skipped_after_error,
        }


class LongContextEvaluator:
    """
    Evaluate any attention architecture via a model forward callable.

    The model interface is intentionally minimal:
      logits = model_fn(input_ids, attention_mask) -> (B, T, V) tensor
    """

    def __init__(
        self,
        config: LongContextBenchmarkConfig | None = None,
        *,
        holdout_samples: list[LongContextSample] | None = None,
    ):
        self.config = config or LongContextBenchmarkConfig()
        self.tokenizer = BenchmarkTokenizer(self.config.vocab_size)
        self.generator = LongContextSampleGenerator(self.config)
        self._holdout_samples = holdout_samples

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def decode_answer_span(
        self,
        logits: torch.Tensor,
        answer_start: int,
        answer_end: int,
    ) -> str:
        """Greedy decode predicted answer token span (causal: logits[t] -> token[t+1])."""
        if answer_end <= answer_start:
            return ""
        logit_start = max(0, answer_start - 1)
        logit_end = max(logit_start, answer_end - 1)
        pred_ids = logits[0, logit_start:logit_end].argmax(dim=-1).tolist()
        return self.tokenizer.decode(pred_ids)

    def score_pointer_sample(
        self,
        pointer_logits: torch.Tensor,
        meta: dict,
    ) -> EvalRecord:
        """Hit@1 on predicted value-token sequence index."""
        if pointer_logits.dim() == 2:
            scores = pointer_logits[0]
        else:
            scores = pointer_logits

        expected_idx = int(meta["pointer_target_index"])
        scoring_mode = meta.get("pointer_scoring_mode", "full_sequence")

        if scoring_mode == "value_slots":
            candidates = [int(i) for i in meta["value_candidate_indices"]]
            pred_slot = int(scores.argmax().item())
            pred_idx = candidates[pred_slot]
            correct = pred_idx == expected_idx
            predicted = str(pred_idx)
            expected = str(expected_idx)
        else:
            question_index = int(meta["question_index"])
            valid_scores = scores[:question_index]
            pred_idx = int(valid_scores.argmax().item())
            correct = pred_idx == expected_idx
            predicted = str(pred_idx)
            expected = str(expected_idx)

        return EvalRecord(
            correct=correct,
            predicted=predicted,
            expected=expected,
            task_type=meta["task_type"],
            context_length=int(meta["context_length"]),
            needle_depth=float(meta["needle_depth"]),
            haystack_mode=str(meta.get("haystack_mode", "")),
        )

    def score_query_only_answer(
        self,
        logits: torch.Tensor,
        meta: dict,
    ) -> EvalRecord:
        """Predict answer token from logits at the final sequence position or pooled head."""
        if logits.dim() == 3:
            scores = logits[0, -1]
        elif logits.dim() == 2:
            scores = logits[0]
        else:
            scores = logits[-1]
        pred_id = int(scores.argmax().item())
        expected = meta["expected_answer"]
        if meta.get("answer_supervision") == "token_id" or meta.get("task_type") == "slot_pointer":
            target_id = int(
                meta.get("query_only_answer_token")
                or meta.get("pointer_target_token")
                or expected
            )
            correct = pred_id == target_id
            predicted = str(pred_id)
            expected = str(target_id)
        else:
            predicted = self.tokenizer.decode([pred_id])
            correct = self._normalize(predicted) == self._normalize(expected)
        return EvalRecord(
            correct=correct,
            predicted=predicted,
            expected=expected,
            task_type=meta["task_type"],
            context_length=int(meta["context_length"]),
            needle_depth=float(meta["needle_depth"]),
            haystack_mode=str(meta.get("haystack_mode", "")),
        )

    def score_sample(
        self,
        logits: torch.Tensor,
        meta: dict,
    ) -> EvalRecord:
        if meta.get("label_mode") == "pointer_index":
            return self.score_pointer_sample(logits, meta)
        if meta.get("label_mode") == "query_only_answer":
            return self.score_query_only_answer(logits, meta)
        answer_start = int(meta["answer_start"])
        answer_end = int(meta["answer_end"])
        predicted = self.decode_answer_span(logits, answer_start, answer_end)
        expected = meta["expected_answer"]
        correct = self._normalize(predicted) == self._normalize(expected)
        return EvalRecord(
            correct=correct,
            predicted=predicted,
            expected=expected,
            task_type=meta["task_type"],
            context_length=int(meta["context_length"]),
            needle_depth=float(meta["needle_depth"]),
            haystack_mode=str(meta.get("haystack_mode", "")),
        )

    def summarize(self, records: list[EvalRecord]) -> EvalSummary:
        total = len(records)
        correct = sum(1 for r in records if r.correct)
        overall = correct / total if total else 0.0

        def _acc(bucket: dict[Any, list[bool]]) -> dict[str, float]:
            out: dict[str, float] = {}
            for key, vals in sorted(bucket.items(), key=lambda x: str(x[0])):
                out[str(key)] = sum(vals) / len(vals) if vals else 0.0
            return out

        by_len: dict[Any, list[bool]] = defaultdict(list)
        by_depth: dict[Any, list[bool]] = defaultdict(list)
        by_task: dict[Any, list[bool]] = defaultdict(list)
        by_cell: dict[Any, list[bool]] = defaultdict(list)
        gate_types = self.config.primary_gate_task_types()
        secondary_types = self.config.secondary_eval_task_types()
        primary_records = [r for r in records if r.task_type in gate_types]
        primary_total = len(primary_records)
        primary_correct = sum(1 for r in primary_records if r.correct)
        primary_acc = primary_correct / primary_total if primary_total else overall

        secondary_records = [r for r in records if r.task_type in secondary_types]
        secondary_total = len(secondary_records)
        secondary_correct = sum(1 for r in secondary_records if r.correct)
        secondary_acc = secondary_correct / secondary_total if secondary_total else 0.0

        for r in records:
            by_len[r.context_length].append(r.correct)
            by_depth[r.needle_depth].append(r.correct)
            by_task[r.task_type].append(r.correct)
            cell = f"{r.task_type}|T={r.context_length}|d={r.needle_depth}"
            by_cell[cell].append(r.correct)

        return EvalSummary(
            overall_accuracy=overall,
            total=total,
            correct=correct,
            primary_gate_accuracy=primary_acc,
            primary_gate_total=primary_total,
            primary_gate_correct=primary_correct,
            secondary_accuracy=secondary_acc,
            secondary_total=secondary_total,
            secondary_correct=secondary_correct,
            pure_niah_accuracy=primary_acc,
            pure_niah_total=primary_total,
            pure_niah_correct=primary_correct,
            by_context_length={int(k): v for k, v in _acc(by_len).items()},
            by_needle_depth=_acc(by_depth),
            by_task_type=_acc(by_task),
            by_cell=_acc(by_cell),
            records=records,
        )

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error" in msg and "memory" in msg

    @torch.no_grad()
    def evaluate_model_fn(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        device: torch.device | None = None,
        max_samples: int | None = None,
        show_progress: bool = True,
        stop_on_first_oom: bool = False,
    ) -> EvalSummary:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = device.type == "cuda"
        samples = self._holdout_samples
        if samples is None:
            samples = get_holdout_grid(self.config)
        dataset = LongContextEvalDataset(self.config, samples=samples)
        non_blocking = pin_memory
        records: list[EvalRecord] = []
        errors: list[EvalError] = []
        skipped = 0
        iterator = dataset
        if show_progress:
            iterator = tqdm(dataset, total=len(dataset), desc="long-context eval (longest first)")
        for i, batch in enumerate(iterator):
            if max_samples is not None and i >= max_samples:
                break
            batch = transfer_batch_to_device(
                batch, device, non_blocking=non_blocking, pin_memory=pin_memory
            )
            meta = batch["meta"][0]
            input_ids = batch["input_ids"]
            attn_mask = batch.get("attention_mask")
            try:
                logits = model_fn(input_ids, attn_mask)
            except RuntimeError as exc:
                if self._is_oom(exc):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    errors.append(
                        EvalError(
                            context_length=int(meta["context_length"]),
                            task_type=str(meta["task_type"]),
                            needle_depth=float(meta["needle_depth"]),
                            error=str(exc),
                        )
                    )
                    if stop_on_first_oom:
                        skipped = max(0, len(dataset) - i - 1)
                        break
                    continue
                raise
            records.append(self.score_sample(logits, meta))
        summary = self.summarize(records)
        summary.errors = errors
        summary.skipped_after_error = skipped
        return summary

    @torch.no_grad()
    def evaluate_module(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        max_samples: int | None = None,
        show_progress: bool = True,
    ) -> EvalSummary:
        device = device or next(model.parameters()).device
        model.eval()

        def _fn(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            out = model(input_ids=input_ids, attn_mask=attn_mask)
            if isinstance(out, dict):
                output_head = getattr(model, "output_head", "lm_token")
                if output_head in ("pointer_index", "pointer_mlp"):
                    return out["pointer_logits"]
                if output_head == "pool_mlp_token":
                    return out["token_logits"]
                return out["logits"]
            return out

        return self.evaluate_model_fn(
            _fn,
            device=device,
            max_samples=max_samples,
            show_progress=show_progress,
            stop_on_first_oom=False,
        )

    @torch.no_grad()
    def benchmark_forward_latency(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        context_length: int = 16384,
        warmup: int = 2,
        runs: int = 5,
        task_type: str | None = None,
    ) -> dict[str, float | int | None]:
        """Time a single eval forward pass at fixed context length."""
        model.eval()
        resolved_task = task_type or (
            self.config.task_types[0] if self.config.task_types else "exact_retrieval"
        )
        haystack_mode = (
            self.config.haystack_modes[0] if self.config.haystack_modes else "random_sentences"
        )
        sample = self.generator.generate_one(
            context_length=context_length,
            needle_depth=0.5,
            task_type=resolved_task,
            haystack_mode=haystack_mode,
            seed=self.config.holdout_seed + context_length,
        )
        input_ids = sample.input_ids.unsqueeze(0).to(device)
        attn_mask = sample.attention_mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).to(device)

        def _forward() -> None:
            out = model(input_ids=input_ids, attn_mask=attn_mask)
            if isinstance(out, dict):
                _ = out["logits"]
            else:
                _ = out

        try:
            for _ in range(warmup):
                _forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings: list[float] = []
            for _ in range(runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _forward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1000.0)
        except RuntimeError as exc:
            if self._is_oom(exc):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return {
                    "context_length": context_length,
                    "latency_ms": None,
                    "tokens_per_sec": None,
                    "error": str(exc),
                }
            raise

        mean_ms = sum(timings) / len(timings)
        tps = (context_length / (mean_ms / 1000.0)) if mean_ms > 0 else None
        return {
            "context_length": context_length,
            "latency_ms": mean_ms,
            "latency_ms_min": min(timings),
            "latency_ms_max": max(timings),
            "tokens_per_sec": tps,
            "warmup": warmup,
            "runs": runs,
        }

    def save_results(self, summary: EvalSummary, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        failures = [
            {
                "expected": r.expected,
                "predicted": r.predicted,
                "task_type": r.task_type,
                "context_length": r.context_length,
                "needle_depth": r.needle_depth,
            }
            for r in summary.records
            if not r.correct
        ]
        with open(output_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures[:200], f, indent=2)
        return summary_path

    def format_tables(self, summary: EvalSummary) -> str:
        lines = [
            "Long-Context Retrieval Benchmark",
            f"Primary gate: {summary.primary_gate_accuracy * 100:.2f}% "
            f"({summary.primary_gate_correct}/{summary.primary_gate_total})",
            f"Overall exact match: {summary.overall_accuracy * 100:.2f}% "
            f"({summary.correct}/{summary.total})",
        ]
        if summary.secondary_total:
            lines.append(
                f"Secondary (distractor): {summary.secondary_accuracy * 100:.2f}% "
                f"({summary.secondary_correct}/{summary.secondary_total})"
            )
        lines.extend(["", "By task type:"])
        for k, v in summary.by_task_type.items():
            lines.append(f"  {k:20s} {v * 100:6.2f}%")
        lines.append("")
        lines.append("By context length:")
        for k, v in sorted(summary.by_context_length.items()):
            lines.append(f"  T={k:6d} {v * 100:6.2f}%")
        lines.append("")
        lines.append("By needle depth:")
        for k, v in sorted(summary.by_needle_depth.items(), key=lambda x: float(x[0])):
            lines.append(f"  depth={k:4s} {v * 100:6.2f}%")
        return "\n".join(lines)
