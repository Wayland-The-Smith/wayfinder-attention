"""Procedural long-context retrieval sample generator."""



from __future__ import annotations



import random

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from dataclasses import dataclass, field



import numpy as np

import torch



from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig

from routing_attention.benchmarks.long_context.haystack import build_haystack_generator

from routing_attention.benchmarks.long_context.synthetic_protocol import (
    assign_non_overlapping_positions,
    hay_chunk_is_clean,
    verify_task_traceable,
)

from routing_attention.benchmarks.long_context.tasks import TaskPayload, generate_task

from routing_attention.benchmarks.long_context.tokenizer import BenchmarkTokenizer





@dataclass

class LongContextSample:

    """One procedurally generated benchmark sample."""



    input_ids: torch.Tensor

    attention_mask: torch.Tensor | None

    labels: torch.Tensor

    answer_start: int

    answer_end: int

    expected_answer: str

    question: str

    task_type: str

    context_length: int

    needle_depth: float

    haystack_mode: str

    metadata: dict = field(default_factory=dict)

    meta_dict: dict = field(default_factory=dict)

    ids_np: np.ndarray | None = None

    labels_np: np.ndarray | None = None

    loss_weights_np: np.ndarray | None = None



    def to_dict(self) -> dict:

        if self.meta_dict:

            return self.meta_dict

        return {

            "expected_answer": self.expected_answer,

            "question": self.question,

            "task_type": self.task_type,

            "context_length": self.context_length,

            "needle_depth": self.needle_depth,

            "haystack_mode": self.haystack_mode,

            "answer_start": self.answer_start,

            "answer_end": self.answer_end,

            "metadata": self.metadata,

        }





def _generate_one_worker(args: tuple[dict, dict]) -> LongContextSample:

    """Picklable worker for parallel eval-grid generation."""

    cfg_dict, kwargs = args

    gen = LongContextSampleGenerator(LongContextBenchmarkConfig.from_dict(cfg_dict))

    return gen.generate_one(**kwargs)





def _filler_text(haystack: str, segments: list[str]) -> str:

    """Haystack with needle segments blanked — used for false-positive collision checks."""

    filler = haystack

    for seg in sorted(segments, key=len, reverse=True):

        filler = filler.replace(seg, "\x00" * len(seg), 1)

    return filler.replace("\x00", "")





def _haystack_has_collision(filler: str, task: TaskPayload) -> bool:

    for token in task.collision_check_strings():

        t = token.strip()

        if t and t in filler:

            return True

    unique_key = task.metadata.get("unique_key")

    if unique_key and f"{unique_key} " in filler:

        return True

    for forbidden in task.metadata.get("filler_forbidden", []):

        if forbidden and forbidden in filler:

            return True

    return False





def _needle_char_depth(haystack: str, segments: list[str], haystack_len: int) -> float:

    """Fractional depth of the primary needle segment center in the haystack."""

    if not segments or haystack_len <= 1:

        return 0.5

    primary = segments[0]

    idx = haystack.find(primary)

    if idx < 0:

        return 0.5

    center = idx + len(primary) // 2

    return center / max(1, haystack_len - 1)


def _needle_spans(haystack: str, segments: list[str]) -> list[tuple[int, int]]:
    """Inclusive start, exclusive end for each needle segment in haystack."""
    spans: list[tuple[int, int]] = []
    for seg in segments:
        if not seg:
            continue
        start = 0
        while True:
            idx = haystack.find(seg, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(seg)))
            start = idx + len(seg)
    return sorted(spans)


def _suffix_start_forbidden(spans: list[tuple[int, int]]) -> set[int]:
    """Positions that would split a needle if the suffix were inserted there."""
    forbidden: set[int] = set()
    for begin, end in spans:
        for pos in range(begin + 1, end):
            forbidden.add(pos)
    return forbidden


def _rightmost_needle_end(haystack: str, segments: list[str]) -> int:
    spans = _needle_spans(haystack, segments)
    if not spans:
        return 0
    return max(end for _, end in spans)


def _gold_needle_start(
    text: str,
    key: str,
    value: str,
    *,
    gold_pattern: str | None = None,
) -> int:
    """Char index of the authoritative needle row (excludes the query token)."""
    if gold_pattern:
        return text.find(gold_pattern)
    return text.find(f"{key} {value}")


def _is_causally_reachable(
    text: str,
    question_start: int,
    key: str,
    value: str,
    *,
    gold_pattern: str | None = None,
) -> bool:
    idx = _gold_needle_start(text, key, value, gold_pattern=gold_pattern)
    return 0 <= idx < question_start


def _query_to_needle_distance(text: str, question_start: int, segments: list[str]) -> int:
    """Min char distance from question_start to any needle char in the final text."""
    best = len(text)
    for seg in segments:
        if not seg:
            continue
        start = 0
        while True:
            idx = text.find(seg, start)
            if idx < 0:
                break
            for pos in range(idx, idx + len(seg)):
                best = min(best, abs(pos - question_start))
            start = idx + len(seg)
    return best if best < len(text) else 0





class LongContextSampleGenerator:

    """

    Generate unlimited unique long-context retrieval samples at runtime.



    Layout (character level):

      [haystack_before][needle(s) in haystack][ Question: Q Answer: A][haystack_after]



    Suffix placement modes (``suffix_placement``):

      ``at_end`` — classic NIAH: query appended after the full haystack.

      ``after_needles`` — query immediately after the rightmost needle (local retrieval).

      ``causal_safe_random`` / ``random`` — random offset in filler, always *after* all needles
      so causal LMs can attend back to the gold row (``random`` is an alias for causal-safe).

    """



    def __init__(self, config: LongContextBenchmarkConfig | None = None):

        self.config = config or LongContextBenchmarkConfig()

        self.tokenizer = BenchmarkTokenizer(self.config.vocab_size)

        self._haystacks = {

            mode: build_haystack_generator(mode, self.config.tinystories_path)

            for mode in self.config.haystack_modes

        }

        self._question_prefix = self.config.question_prefix

        self._question_prefix_len = len(self._question_prefix)

        self._answer_prefix = self.config.answer_prefix

        self._answer_prefix_len = len(self._answer_prefix)

        self._depths = tuple(self.config.needle_depths)

        self._tasks = tuple(self.config.task_types)

        self._modes = tuple(self.config.haystack_modes)



    def _rng(self, seed: int | None = None) -> random.Random:

        base = self.config.seed if seed is None else seed

        return random.Random(base)



    def _combined_needle(self, needles: list[str]) -> str:

        return " ".join(needles) if needles else ""



    def _insert_needles(self, haystack: str, combined_needle: str, depth: float) -> str:

        if not combined_needle:

            return haystack

        if not haystack:

            return combined_needle

        pos = int(depth * max(0, len(haystack) - 1))

        pos = max(0, min(pos, len(haystack)))

        return haystack[:pos] + combined_needle + " " + haystack[pos:]



    def _pick_suffix_start_safe(
        self,
        haystack: str,
        segments: list[str],
        rng: random.Random,
    ) -> int | None:
        """Pick suffix offset in filler only — never inside a needle segment."""
        haystack_total = len(haystack)
        cfg_min = int(self.config.min_haystack_side_chars)
        min_side = min(cfg_min, max(8, haystack_total // 8))
        spans = _needle_spans(haystack, segments)
        forbidden = _suffix_start_forbidden(spans)

        lo = max(min_side, int(haystack_total * float(self.config.suffix_depth_min)))
        hi = min(haystack_total - min_side, int(haystack_total * float(self.config.suffix_depth_max)))
        if hi < lo:
            lo = min_side
            hi = max(min_side, haystack_total - min_side)

        candidates = [
            pos
            for pos in range(lo, hi + 1)
            if pos not in forbidden and pos >= min_side and (haystack_total - pos) >= min_side
        ]
        if not candidates:
            candidates = [
                pos
                for pos in range(haystack_total + 1)
                if pos not in forbidden
                and pos >= min_side
                and (haystack_total - pos) >= min_side
            ]
        if not candidates:
            return None
        return rng.choice(candidates)

    def _pick_suffix_start_at_end(
        self,
        haystack: str,
        segments: list[str],
        rng: random.Random,
    ) -> int:
        """Classic NIAH: append query suffix after the entire haystack."""
        del segments, rng
        return len(haystack)

    def _pick_suffix_start_causal_safe_random(
        self,
        haystack: str,
        segments: list[str],
        rng: random.Random,
    ) -> int | None:
        """Random suffix offset in filler, constrained to positions at/after all needles."""
        haystack_total = len(haystack)
        cfg_min = int(self.config.min_haystack_side_chars)
        min_side = min(cfg_min, max(8, haystack_total // 8))
        forbidden = _suffix_start_forbidden(_needle_spans(haystack, segments))
        right_end = _rightmost_needle_end(haystack, segments)

        lo = max(min_side, int(haystack_total * float(self.config.suffix_depth_min)), right_end)
        hi = min(haystack_total - min_side, int(haystack_total * float(self.config.suffix_depth_max)))
        if hi < lo:
            # Needle ends below the depth band — still place suffix after needles (causal-safe).
            lo = max(right_end, min_side)
            hi = max(lo, haystack_total - min_side)

        candidates = [
            pos
            for pos in range(lo, hi + 1)
            if pos not in forbidden and pos >= min_side and (haystack_total - pos) >= min_side
        ]
        if not candidates:
            candidates = [
                pos
                for pos in range(right_end, haystack_total + 1)
                if pos not in forbidden
                and pos >= min(8, min_side)
                and (haystack_total - pos) >= min(8, min_side)
            ]
        if not candidates:
            return self._pick_suffix_start_at_end(haystack, segments, rng)
        return rng.choice(candidates)

    def _pick_suffix_start_after_needles(
        self,
        haystack: str,
        segments: list[str],
        rng: random.Random,
    ) -> int | None:
        """Place query suffix immediately after the rightmost needle segment."""
        haystack_total = len(haystack)
        cfg_min = int(self.config.min_haystack_side_chars)
        min_side = min(cfg_min, max(8, haystack_total // 8))
        spans = _needle_spans(haystack, segments)
        forbidden = _suffix_start_forbidden(spans)
        if not spans:
            return self._pick_suffix_start_safe(haystack, segments, rng)
        right_end = max(end for _, end in spans)
        gap_max = max(0, int(self.config.suffix_after_needles_gap_max))
        gap = rng.randint(0, gap_max) if gap_max > 0 else 0
        pos = min(right_end + gap, haystack_total)
        if pos in forbidden:
            pos = right_end
        if pos in forbidden:
            return self._pick_suffix_start_safe(haystack, segments, rng)
        if pos < min_side or (haystack_total - pos) < min_side:
            relaxed = [
                p
                for p in range(max(0, right_end), haystack_total + 1)
                if p not in forbidden
                and p >= min(8, min_side)
                and (haystack_total - p) >= min(8, min_side)
            ]
            if not relaxed:
                return self._pick_suffix_start_safe(haystack, segments, rng)
            return rng.choice(relaxed)
        return pos

    def _pick_suffix_start(
        self,
        haystack: str,
        segments: list[str],
        rng: random.Random,
    ) -> int | None:
        placement = self.config.suffix_placement
        if placement == "after_needles":
            return self._pick_suffix_start_after_needles(haystack, segments, rng)
        if placement == "at_end":
            return self._pick_suffix_start_at_end(haystack, segments, rng)
        if placement in ("causal_safe_random", "random"):
            return self._pick_suffix_start_causal_safe_random(haystack, segments, rng)
        return self._pick_suffix_start_safe(haystack, segments, rng)

    def _build_haystack_with_needles(

        self,

        *,

        haystack_len: int,

        segments: list[str],

        needle_depth: float,

        haystack_mode: str,

        scatter: bool,

        rng: random.Random,

        task: TaskPayload | None = None,

    ) -> tuple[str, list[float]]:

        if haystack_len < 32:

            raise ValueError(f"haystack_len too small: {haystack_len}")



        if (
            getattr(self.config, "benchmark_family", "") == "synthetic"
            and scatter
            and len(segments) > 1
        ):
            return self._build_hop_first_scattered(
                haystack_len,
                segments,
                task=task,
                haystack_mode=haystack_mode,
                rng=rng,
            )

        if scatter and len(segments) > 1:

            return self._build_scattered(haystack_len, segments, needle_depth, haystack_mode, rng)

        combined = self._combined_needle(segments)

        needle_len = len(combined)

        sep_len = 1 if combined else 0

        reserve = needle_len + sep_len

        if reserve >= haystack_len:

            raise ValueError(f"needle length {reserve} exceeds haystack budget {haystack_len}")

        base_len = haystack_len - reserve

        haystack = self._haystacks[haystack_mode].generate(rng, base_len)

        body = self._insert_needles(haystack, combined, needle_depth)

        if len(body) < haystack_len:

            need = haystack_len - len(body)

            body += self._haystacks[haystack_mode].generate(rng, need)

        if len(body) != haystack_len:

            raise RuntimeError(f"haystack length {len(body)} != {haystack_len}")

        depth = _needle_char_depth(body, segments, haystack_len)

        return body, [depth]



    def _build_scattered(

        self,

        haystack_len: int,

        segments: list[str],

        needle_depth: float,

        haystack_mode: str,

        rng: random.Random,

    ) -> tuple[str, list[float]]:

        total_reserve = sum(len(seg) + 1 for seg in segments)

        if total_reserve >= haystack_len:

            raise ValueError(f"combined needle length {total_reserve} exceeds haystack {haystack_len}")

        base_len = haystack_len - total_reserve

        haystack = self._haystacks[haystack_mode].generate(rng, base_len)



        center = int(needle_depth * max(0, base_len - 1))

        spread = max(1, base_len // max(4, 2 * len(segments)))

        positions: list[int] = []

        for i in range(len(segments)):

            offset = (i - len(segments) // 2) * spread + rng.randint(-spread // 2, spread // 2)

            pos = max(0, min(base_len - 1, center + offset))

            positions.append(pos)

        positions.sort()

        for i in range(1, len(positions)):

            if positions[i] < positions[i - 1]:

                positions[i] = positions[i - 1]



        parts: list[str] = []
        cursor = 0
        for insert_pos, segment in zip(positions, segments):
            insert_pos = max(cursor, min(insert_pos, base_len))
            parts.append(haystack[cursor:insert_pos])
            parts.append(segment + " ")
            cursor = insert_pos
        parts.append(haystack[cursor:])
        body = "".join(parts)
        if len(body) != haystack_len:
            raise RuntimeError(f"scattered haystack length {len(body)} != {haystack_len}")
        depths = []
        for segment in segments:
            idx = body.find(segment)
            if idx < 0:
                depths.append(float(needle_depth))
            else:
                center = idx + len(segment) / 2.0
                depths.append(center / max(1, haystack_len - 1))
        return body, depths



    def _generate_clean_hay_chunk(
        self,
        length: int,
        task: TaskPayload,
        haystack_mode: str,
        rng: random.Random,
        *,
        max_attempts: int = 32,
    ) -> str:
        if length <= 0:
            return ""
        for _ in range(max_attempts):
            chunk = self._haystacks[haystack_mode].generate(rng, length)
            if hay_chunk_is_clean(chunk, task):
                return chunk
        raise ValueError("failed to generate collision-free hay chunk")

    def _placement_bounds(self, haystack_len: int) -> tuple[int, int]:
        pos_min = max(0, int(self.config.scatter_placement_min))
        pos_max = self.config.scatter_placement_max
        if pos_max is None:
            pos_max = haystack_len
        pos_max = min(haystack_len, max(pos_min, int(pos_max)))
        return pos_min, pos_max

    def _build_hop_first_scattered(
        self,
        haystack_len: int,
        segments: list[str],
        *,
        task: TaskPayload,
        haystack_mode: str,
        rng: random.Random,
    ) -> tuple[str, list[float]]:
        """Hop-first protocol: collision-free hay, then uniformly scattered needles."""
        if task is None:
            raise ValueError("hop-first scattered layout requires TaskPayload")
        if haystack_len < 32:
            raise ValueError(f"haystack_len too small: {haystack_len}")

        unit_lens = [len(seg) + 1 for seg in segments]
        total_needle = sum(unit_lens)
        if total_needle >= haystack_len:
            raise ValueError(f"needle length {total_needle} exceeds haystack budget {haystack_len}")

        pos_min, pos_max = self._placement_bounds(haystack_len)
        positions = assign_non_overlapping_positions(
            rng,
            seg_lens=unit_lens,
            pos_min=pos_min,
            pos_max=pos_max,
        )
        if positions is None:
            raise ValueError("no valid scattered needle positions in placement bounds")

        order = sorted(range(len(segments)), key=lambda i: positions[i])
        gap_lengths: list[int] = []
        cursor = 0
        for idx in order:
            pos = positions[idx]
            gap_lengths.append(max(0, pos - cursor))
            cursor = pos + unit_lens[idx]
        gap_lengths.append(max(0, haystack_len - cursor))

        gap_chunks = [
            self._generate_clean_hay_chunk(n, task, haystack_mode, rng)
            for n in gap_lengths
        ]

        parts: list[str] = []
        for gi, idx in enumerate(order):
            parts.append(gap_chunks[gi])
            parts.append(segments[idx] + " ")
        parts.append(gap_chunks[-1])
        body = "".join(parts)
        if len(body) != haystack_len:
            raise RuntimeError(f"hop-first haystack length {len(body)} != {haystack_len}")

        if not verify_task_traceable(body, task):
            raise ValueError("assembled haystack failed task trace verification")

        depths = []
        for seg in segments:
            idx = body.find(seg)
            if idx < 0:
                depths.append(0.5)
            else:
                center = idx + len(seg) / 2.0
                depths.append(center / max(1, haystack_len - 1))
        return body, depths



    def _suffix_for_task(self, task: TaskPayload) -> str:
        question = f"{self._question_prefix}{task.question}"
        if not self.config.include_answer_in_suffix:
            return question
        return f"{question}{self._answer_prefix}{task.expected_answer}"



    def _char_offsets(

        self,

        *,

        suffix_start: int,

        question: str,

        answer_len: int,

    ) -> tuple[int, int, int]:

        """Return (question_start, answer_start, answer_end) in token/char indices."""

        question_start = suffix_start + self._question_prefix_len

        answer_start = question_start + len(question) + self._answer_prefix_len

        answer_end = answer_start + answer_len

        return question_start, answer_start, answer_end



    def _tokenize_sample(

        self,

        text: str,

        context_length: int,

        *,

        question_start: int,

        answer_start: int,

        answer_end: int,

        label_mode: str,

        answer_loss_weight: float,

        answer_supervision_mask: tuple[bool, ...] | None,

        query_only_answer_token: int | None = None,

    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int, int, np.ndarray, np.ndarray, np.ndarray]:

        """Char tokenizer is 1:1 — token indices match char indices."""

        ids_arr = self.tokenizer.encode_array(text)

        if ids_arr.shape[0] != context_length:

            raise RuntimeError(

                f"token length {ids_arr.shape[0]} != context_length {context_length}"

            )



        if label_mode not in ("answer_only", "query_only_answer"):

            raise ValueError(

                f"train_label_mode={label_mode!r} is not supported; "

                "only answer_only or query_only_answer may be supervised"

            )

        if label_mode == "query_only_answer":
            if query_only_answer_token is None:
                raise ValueError("query_only_answer_token is required for query_only_answer mode")
            labels_arr = ids_arr.copy()
            labels_arr[:] = -100
            weights_arr = np.zeros(context_length, dtype=np.float32)
            pos = context_length - 1
            labels_arr[pos] = int(query_only_answer_token)
            weights_arr[pos] = float(answer_loss_weight)
            ids_np = ids_arr.astype(np.int64, copy=False)
            labels_np = labels_arr.astype(np.int64, copy=False)
            input_ids = torch.from_numpy(ids_np)
            labels = torch.from_numpy(labels_np)
            has_pad = bool((ids_arr == self.tokenizer.pad_token_id).any())
            attention_mask = None if not has_pad else (input_ids != self.tokenizer.pad_token_id).long()
            return (
                input_ids,
                attention_mask,
                labels,
                answer_start,
                answer_end,
                ids_np,
                labels_np,
                weights_arr,
            )

        if label_mode != "answer_only":
            raise ValueError(f"unexpected label_mode {label_mode!r}")

        answer_start = min(answer_start, context_length - 1)

        answer_end = min(context_length, answer_end)

        if answer_end <= answer_start:

            raise RuntimeError(

                f"empty answer span [{answer_start}, {answer_end}) for supervision"

            )



        labels_arr = ids_arr.copy()

        labels_arr[:] = -100

        weights_arr = np.zeros(context_length, dtype=np.float32)

        ans_len = answer_end - answer_start

        if answer_supervision_mask is not None and len(answer_supervision_mask) != ans_len:

            raise ValueError(

                f"answer_supervision_mask length {len(answer_supervision_mask)} "

                f"!= answer span {ans_len}"

            )

        for i, pos in enumerate(range(answer_start, answer_end)):

            if answer_supervision_mask is not None and not answer_supervision_mask[i]:

                continue

            labels_arr[pos] = ids_arr[pos]

            weights_arr[pos] = float(answer_loss_weight)



        ids_np = ids_arr.astype(np.int64, copy=False)

        labels_np = labels_arr.astype(np.int64, copy=False)

        loss_weights_np = weights_arr

        input_ids = torch.from_numpy(ids_np)

        labels = torch.from_numpy(labels_np)

        has_pad = bool((ids_arr == self.tokenizer.pad_token_id).any())

        attention_mask = None if not has_pad else (input_ids != self.tokenizer.pad_token_id).long()

        return (

            input_ids,

            attention_mask,

            labels,

            answer_start,

            answer_end,

            ids_np,

            labels_np,

            loss_weights_np,

        )



    def _assemble_sample(

        self,

        *,

        context_length: int,

        needle_depth: float,

        task: TaskPayload,

        task_type: str,

        haystack_mode: str,

        rng: random.Random,

        label_mode: str,

    ) -> LongContextSample:

        suffix = self._suffix_for_task(task)

        if len(suffix) >= context_length:

            raise ValueError(

                f"suffix length {len(suffix)} >= context_length {context_length} "

                f"for task {task_type}"

            )

        haystack_total = context_length - len(suffix)

        segments = list(task.needle_segments)

        scatter = bool(self.config.scatter_multi_needles) and len(segments) > 1

        haystack, segment_depths = self._build_haystack_with_needles(

            haystack_len=haystack_total,

            segments=segments,

            needle_depth=needle_depth,

            haystack_mode=haystack_mode,

            scatter=scatter,

            rng=rng,

            task=task,

        )



        filler = _filler_text(haystack, segments)

        if _haystack_has_collision(filler, task):

            raise ValueError("haystack filler contains answer or collision token")



        suffix_start = self._pick_suffix_start(haystack, segments, rng)

        if suffix_start is None:

            raise ValueError("no valid suffix position outside needle spans")



        text = haystack[:suffix_start] + suffix + haystack[suffix_start:]

        if len(text) != context_length:

            raise RuntimeError(f"text length {len(text)} != context_length {context_length}")



        answer_len = len(task.expected_answer)

        effective_label_mode = label_mode
        if not self.config.include_answer_in_suffix:
            effective_label_mode = "query_only_answer"

        if self.config.include_answer_in_suffix:
            question_start, answer_start, answer_end = self._char_offsets(
                suffix_start=suffix_start,
                question=task.question,
                answer_len=answer_len,
            )
        else:
            question_start = suffix_start + self._question_prefix_len
            answer_start = context_length - 1
            answer_end = context_length

        query_only_token: int | None = None
        if effective_label_mode == "query_only_answer":
            answer_ids = self.tokenizer.encode(task.expected_answer)[
                : int(self.config.max_answer_chars)
            ]
            if not answer_ids:
                raise ValueError("expected_answer tokenization produced no ids")
            query_only_token = int(answer_ids[0])

        if answer_end > context_length:

            raise RuntimeError(f"answer_end {answer_end} exceeds context_length {context_length}")

        key = task.metadata.get("key")
        value = task.metadata.get("value")
        gold_pattern = task.metadata.get("gold_needle")
        if key is not None and value is not None and not _is_causally_reachable(
            text,
            question_start,
            str(key),
            str(value),
            gold_pattern=str(gold_pattern) if gold_pattern else None,
        ):
            raise ValueError(
                f"gold needle {key!r} {value!r} not causally reachable from query at {question_start}"
            )

        report_depth = float(sum(segment_depths) / len(segment_depths))

        q_needle_dist = _query_to_needle_distance(text, question_start, segments)

        needle_centers = []
        for seg in segments:
            idx = text.find(seg)
            if idx >= 0:
                needle_centers.append((idx + len(seg) / 2.0) / max(1, context_length - 1))
        needle_depth_final = (
            float(sum(needle_centers) / len(needle_centers)) if needle_centers else report_depth
        )



        (

            input_ids,

            attention_mask,

            labels,

            answer_start,

            answer_end,

            ids_np,

            labels_np,

            loss_weights_np,

        ) = self._tokenize_sample(

            text,

            context_length,

            question_start=question_start,

            answer_start=answer_start,

            answer_end=answer_end,

            label_mode=effective_label_mode,

            answer_loss_weight=float(self.config.answer_loss_weight),

            answer_supervision_mask=task.answer_supervision_mask,

            query_only_answer_token=query_only_token,

        )



        meta_dict = {

            "expected_answer": task.expected_answer,

            "question": task.question,

            "task_type": task_type,

            "context_length": context_length,

            "needle_depth": report_depth,

            "haystack_mode": haystack_mode,

            "answer_start": answer_start,

            "answer_end": answer_end,

            "question_start": question_start,

            "suffix_start": suffix_start,

            "suffix_depth": suffix_start / max(1, haystack_total),

            "query_to_needle_distance": q_needle_dist,

            "needle_depth_final": needle_depth_final,

            "label_mode": effective_label_mode,
            "include_answer_in_suffix": bool(self.config.include_answer_in_suffix),

            "scatter_needles": scatter,

            "segment_depths": segment_depths,

            "metadata": task.metadata,

            "needle_segments": segments,

        }

        if effective_label_mode == "query_only_answer":
            meta_dict["question_index"] = context_length - 1
            if query_only_token is not None:
                meta_dict["query_only_answer_token"] = int(query_only_token)



        return LongContextSample(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels,

            answer_start=answer_start,

            answer_end=answer_end,

            expected_answer=task.expected_answer,

            question=task.question,

            task_type=task_type,

            context_length=context_length,

            needle_depth=report_depth,

            haystack_mode=haystack_mode,

            metadata=task.metadata,

            meta_dict=meta_dict,

            ids_np=ids_np,

            labels_np=labels_np,

            loss_weights_np=loss_weights_np,

        )



    def _assemble_slot_pointer_sample(
        self,
        *,
        context_length: int,
        num_quads: int,
        rng: random.Random,
        seed: int | None,
        task_type: str = "slot_pointer",
    ) -> LongContextSample:
        from routing_attention.benchmarks.long_context.slot_pointer import (
            generate_slot_pointer_tokens,
            slot_pointer_value_slot_labels,
            verify_slot_pointer_tokens,
        )

        slot = generate_slot_pointer_tokens(
            rng,
            context_length=context_length,
            num_quads=num_quads,
            placement=str(getattr(self.config, "slot_quad_placement", "random")),
            enforce_unique_semantics=bool(
                getattr(self.config, "slot_enforce_unique_semantics", False)
            ),
        )
        verify_slot_pointer_tokens(
            slot.tokens,
            question_index=slot.question_index,
            expected_value_index=slot.target_value_index,
        )

        ids_arr = np.asarray(slot.tokens, dtype=np.int32)
        if ids_arr.shape[0] != context_length:
            raise RuntimeError(
                f"slot_pointer length {ids_arr.shape[0]} != context_length {context_length}"
            )

        label_mode = str(getattr(self.config, "train_label_mode", "answer_only") or "answer_only")
        if label_mode not in ("answer_only", "query_only_answer", "pointer_index"):
            raise ValueError(
                f"slot_pointer train_label_mode must be answer_only, query_only_answer, or "
                f"pointer_index, got {label_mode!r}"
            )
        # Legacy pointer heads use pointer_index supervision (labels masked; CE in pointer head).
        if label_mode == "answer_only":
            label_mode = "pointer_index"

        labels_arr = np.full(context_length, -100, dtype=np.int32)
        weights_arr = np.zeros(context_length, dtype=np.float32)
        target = slot.target_value_index
        question_index = slot.question_index
        value_candidate_indices, pointer_target_slot = slot_pointer_value_slot_labels(
            slot.quads,
            target,
        )
        if label_mode == "query_only_answer":
            answer_weight = float(getattr(self.config, "answer_loss_weight", 1.0))
            labels_arr[question_index] = int(slot.target_value_token)
            weights_arr[question_index] = answer_weight

        needle_depth = target / max(1, context_length - 1)

        ids_np = ids_arr.astype(np.int64, copy=False)
        labels_np = labels_arr.astype(np.int64, copy=False)
        input_ids = torch.from_numpy(ids_np)
        labels = torch.from_numpy(labels_np)

        meta_dict = {
            "expected_answer": str(slot.target_value_token),
            "question": str(slot.query_addr_token),
            "task_type": task_type,
            "context_length": context_length,
            "needle_depth": needle_depth,
            "haystack_mode": "slot_pointer",
            "answer_start": target,
            "answer_end": target + 1,
            "question_start": question_index,
            "question_index": question_index,
            "pointer_target_index": target,
            "pointer_target_slot": pointer_target_slot,
            "value_candidate_indices": value_candidate_indices,
            "pointer_scoring_mode": "value_slots",
            "slot_quad_placement": str(getattr(self.config, "slot_quad_placement", "random")),
            "slot_enforce_unique_semantics": bool(
                getattr(self.config, "slot_enforce_unique_semantics", False)
            ),
            "benchmark_variant": str(getattr(self.config, "benchmark_variant", "") or ""),
            "pointer_target_token": slot.target_value_token,
            "query_addr_token": slot.query_addr_token,
            "num_slot_quads": num_quads,
            "label_mode": label_mode,
            "include_answer_in_suffix": False,
            "query_only_answer_token": int(slot.target_value_token),
            "answer_supervision": "token_id",
            "scatter_needles": True,
            "metadata": {
                "protocol": (
                    "slot_pointer_query_only_v1"
                    if label_mode == "query_only_answer"
                    else "slot_pointer_v1"
                ),
                "num_quads": num_quads,
                "query_addr_token": slot.query_addr_token,
                "target_value_index": target,
                "target_value_token": slot.target_value_token,
                "quads": [
                    {
                        "addr_token": q.addr_token,
                        "value_token": q.value_token,
                        "start_index": q.start_index,
                        "value_index": q.value_index,
                    }
                    for q in slot.quads
                ],
            },
            "seed": seed,
        }

        return LongContextSample(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
            answer_start=target,
            answer_end=target + 1,
            expected_answer=str(slot.target_value_token),
            question=str(slot.query_addr_token),
            task_type=task_type,
            context_length=context_length,
            needle_depth=needle_depth,
            haystack_mode="slot_pointer",
            metadata=meta_dict["metadata"],
            meta_dict=meta_dict,
            ids_np=ids_np,
            labels_np=labels_np,
            loss_weights_np=weights_arr,
        )



    def generate_one(

        self,

        *,

        context_length: int,

        needle_depth: float,

        task_type: str,

        haystack_mode: str | None = None,

        seed: int | None = None,

        label_mode: str | None = None,

    ) -> LongContextSample:

        if context_length < 64:

            raise ValueError("context_length must be >= 64")



        lm = label_mode or self.config.train_label_mode

        if lm not in ("answer_only", "query_only_answer"):
            raise ValueError(
                f"generate_one label_mode must be answer_only or query_only_answer, got {lm!r}"
            )



        base_seed = self.config.seed if seed is None else seed

        max_attempts = int(self.config.generation_max_attempts)



        for attempt in range(max_attempts):

            rng = random.Random(base_seed + attempt * 7919)

            if task_type == "slot_pointer":
                num_quads = int(self.config.num_slot_quads)
                try:
                    return self._assemble_slot_pointer_sample(
                        context_length=context_length,
                        num_quads=num_quads,
                        rng=rng,
                        seed=base_seed,
                        task_type=task_type,
                    )
                except ValueError:
                    continue

            mode = haystack_mode if haystack_mode is not None else rng.choice(self.config.haystack_modes)

            if mode not in self._haystacks:

                raise ValueError(f"Unknown haystack_mode '{mode}'")



            task_kwargs = {

                "num_distractors": self.config.num_distractors,

                "num_multi_keys": self.config.num_multi_keys,

                "num_needles_multi": self.config.num_needles_multi,

                "max_answer_chars": self.config.max_answer_chars,

                "synthetic_hop_count": self.config.synthetic_hop_count,
                "synthetic_hop_count_min": self.config.synthetic_hop_count_min,
                "synthetic_hop_count_max": self.config.synthetic_hop_count_max,
                "synthetic_decoy_keys": self.config.synthetic_decoy_keys,
                "synthetic_decoy_addrs": self.config.synthetic_decoy_addrs,
                "synthetic_fake_ptrs": self.config.synthetic_fake_ptrs,
                "num_kv_pairs": self.config.num_kv_pairs,
                "num_queries": self.config.num_queries,
                "mqar_supervise_all_queries": self.config.mqar_supervise_all_queries,
                "answer_digit_width": self.config.answer_digit_width,
                "synthetic_conflict_rows": self.config.synthetic_conflict_rows,
                "num_slot_quads": self.config.num_slot_quads,

            }

            if getattr(self.config, "benchmark_family", "nl") == "synthetic":

                from routing_attention.benchmarks.long_context.tasks_synthetic import (

                    generate_synthetic_task,

                )

                task = generate_synthetic_task(task_type, rng, task_kwargs)

            else:

                task = generate_task(task_type, rng, task_kwargs)

            try:

                return self._assemble_sample(

                    context_length=context_length,

                    needle_depth=needle_depth,

                    task=task,

                    task_type=task_type,

                    haystack_mode=mode,

                    rng=rng,

                    label_mode=lm,

                )

            except ValueError:

                continue



        raise RuntimeError(

            f"Failed to generate valid sample for {task_type!r} at T={context_length} "

            f"after {max_attempts} attempts (collision or layout)"

        )



    def generate_task_payload(
        self,
        *,
        task_type: str,
        haystack_mode: str | None = None,
        seed: int | None = None,
    ) -> tuple[TaskPayload, str]:
        """Sample task content (needles + query + answer) without layout."""
        base_seed = self.config.seed if seed is None else seed
        rng = random.Random(base_seed)
        mode = haystack_mode if haystack_mode is not None else rng.choice(self.config.haystack_modes)
        if mode not in self._haystacks:
            raise ValueError(f"Unknown haystack_mode '{mode}'")

        task_kwargs = {
            "num_distractors": self.config.num_distractors,
            "num_multi_keys": self.config.num_multi_keys,
            "num_needles_multi": self.config.num_needles_multi,
            "max_answer_chars": self.config.max_answer_chars,
            "synthetic_hop_count": self.config.synthetic_hop_count,
            "synthetic_hop_count_min": self.config.synthetic_hop_count_min,
            "synthetic_hop_count_max": self.config.synthetic_hop_count_max,
            "synthetic_decoy_keys": self.config.synthetic_decoy_keys,
            "synthetic_decoy_addrs": self.config.synthetic_decoy_addrs,
            "synthetic_fake_ptrs": self.config.synthetic_fake_ptrs,
            "num_kv_pairs": self.config.num_kv_pairs,
            "num_queries": self.config.num_queries,
            "mqar_supervise_all_queries": self.config.mqar_supervise_all_queries,
            "answer_digit_width": self.config.answer_digit_width,
            "synthetic_conflict_rows": self.config.synthetic_conflict_rows,
        }
        if getattr(self.config, "benchmark_family", "nl") == "synthetic":
            from routing_attention.benchmarks.long_context.tasks_synthetic import (
                generate_synthetic_task,
            )

            task = generate_synthetic_task(task_type, rng, task_kwargs)
        else:
            task = generate_task(task_type, rng, task_kwargs)
        return task, mode

    def assemble_from_task(
        self,
        *,
        task: TaskPayload,
        task_type: str,
        context_length: int,
        needle_depth: float,
        haystack_mode: str,
        seed: int,
        label_mode: str | None = None,
    ) -> LongContextSample:
        """Lay out a fixed task payload with a new random needle scatter (placement only)."""
        if context_length < 64:
            raise ValueError("context_length must be >= 64")
        lm = label_mode or self.config.train_label_mode
        if lm != "answer_only":
            raise ValueError(f"assemble_from_task label_mode must be answer_only, got {lm!r}")

        max_attempts = int(self.config.generation_max_attempts)
        for attempt in range(max_attempts):
            rng = random.Random(seed + attempt * 7919)
            try:
                return self._assemble_sample(
                    context_length=context_length,
                    needle_depth=needle_depth,
                    task=task,
                    task_type=task_type,
                    haystack_mode=haystack_mode,
                    rng=rng,
                    label_mode=lm,
                )
            except ValueError:
                continue
        raise RuntimeError(
            f"Failed to assemble layout for {task_type!r} at T={context_length} "
            f"after {max_attempts} attempts (collision or layout)"
        )



    def _grid_jobs(

        self,

        ctx: list[int],

        depths: list[float],

        tasks: list[str],

        modes: list[str],

        samples_per_cell: int,

        seed0: int,

    ) -> list[tuple[dict, dict]]:

        cfg_dict = self.config.to_dict()

        jobs: list[tuple[dict, dict]] = []

        idx = 0

        for task_type in tasks:

            for mode in modes:

                for length in ctx:

                    for depth in depths:

                        for _ in range(samples_per_cell):

                            jobs.append(

                                (

                                    cfg_dict,

                                    {

                                        "context_length": length,

                                        "needle_depth": depth,

                                        "task_type": task_type,

                                        "haystack_mode": mode,

                                        "seed": seed0 + idx,

                                        "label_mode": "answer_only",

                                    },

                                )

                            )

                            idx += 1

        return jobs



    def generate_grid(

        self,

        *,

        context_lengths: list[int] | None = None,

        needle_depths: list[float] | None = None,

        task_types: list[str] | None = None,

        haystack_modes: list[str] | None = None,

        samples_per_cell: int = 1,

        base_seed: int | None = None,

        num_workers: int = 0,

    ) -> list[LongContextSample]:

        ctx = sorted(context_lengths or self.config.context_lengths, reverse=True)

        depths = needle_depths or self.config.needle_depths

        tasks = task_types or self.config.task_types

        modes = haystack_modes or self.config.haystack_modes

        seed0 = self.config.seed if base_seed is None else base_seed

        workers = num_workers if num_workers > 0 else self.config.eval_grid_workers

        jobs = self._grid_jobs(ctx, depths, tasks, modes, samples_per_cell, seed0)



        if workers > 1 and len(jobs) > 1:

            chunk = max(1, len(jobs) // (workers * 4))

            try:

                with ProcessPoolExecutor(max_workers=workers) as pool:

                    samples = list(pool.map(_generate_one_worker, jobs, chunksize=chunk))

            except Exception:

                with ThreadPoolExecutor(max_workers=workers) as pool:

                    samples = list(pool.map(_generate_one_worker, jobs, chunksize=chunk))

        else:

            samples = [self.generate_one(**job[1]) for job in jobs]



        return sorted(

            samples,

            key=lambda s: (-s.context_length, s.task_type, s.haystack_mode, -s.needle_depth),

        )


