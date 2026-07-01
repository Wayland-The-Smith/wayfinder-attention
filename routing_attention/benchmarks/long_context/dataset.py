"""PyTorch datasets wrapping the procedural long-context generator."""



from __future__ import annotations



import queue

import random

import threading

from typing import Iterator



import numpy as np

import torch

from torch.utils.data import DataLoader, IterableDataset



from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig

from routing_attention.benchmarks.long_context.generator import LongContextSample, LongContextSampleGenerator

from routing_attention.benchmarks.long_context.tasks import TaskPayload

from routing_attention.benchmarks.long_context.holdout import get_holdout_grid






def _stack_samples(samples: list[LongContextSample]) -> dict[str, torch.Tensor | list | None]:

    """Stack samples as CPU tensors (pinning happens in main process before H2D copy)."""

    ids_np = np.stack(

        [s.ids_np if s.ids_np is not None else s.input_ids.numpy() for s in samples],

        dtype=np.int64,

    )

    labels_np = np.stack(

        [s.labels_np if s.labels_np is not None else s.labels.numpy() for s in samples],

        dtype=np.int64,

    )

    weights_np = np.stack(

        [
            s.loss_weights_np
            if s.loss_weights_np is not None
            else np.ones(s.input_ids.shape[0], dtype=np.float32)
            for s in samples
        ],

        dtype=np.float32,

    )

    out: dict[str, torch.Tensor | list | None] = {

        "input_ids": torch.from_numpy(ids_np),

        "labels": torch.from_numpy(labels_np),

        "loss_weights": torch.from_numpy(weights_np),

        "meta": [s.meta_dict or s.to_dict() for s in samples],

    }

    if any(s.attention_mask is not None for s in samples):

        masks_np = np.stack(

            [

                s.attention_mask.numpy()

                if s.attention_mask is not None

                else np.ones(s.input_ids.shape[0], dtype=np.int64)

                for s in samples

            ],

            dtype=np.int64,

        )

        out["attention_mask"] = torch.from_numpy(masks_np)

    else:

        out["attention_mask"] = None

    return out





def _sample_to_eval_batch(sample: LongContextSample) -> dict:
    """Pre-built single-sample eval batch (CPU tensors)."""
    if sample.ids_np is not None and sample.labels_np is not None:
        input_ids = torch.from_numpy(sample.ids_np).unsqueeze(0)
        labels = torch.from_numpy(sample.labels_np).unsqueeze(0)
    else:
        input_ids = sample.input_ids.unsqueeze(0)
        labels = sample.labels.unsqueeze(0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "meta": [sample.meta_dict or sample.to_dict()],
        "attention_mask": (
            sample.attention_mask.unsqueeze(0) if sample.attention_mask is not None else None
        ),
    }





class LongContextTrainDataset(IterableDataset):

    """

    Infinite step-based training stream at a **fixed** context length (no epochs).



    Uses ``config.seed`` only — never ``holdout_seed``.

    """



    def __init__(

        self,

        config: LongContextBenchmarkConfig | None = None,

        batch_size: int = 1,

        include_mask: bool = True,

        train_context_length: int | None = None,

    ):

        if train_context_length is None:

            raise ValueError(

                "train_context_length is required — suite trains one fixed T per sub-experiment"

            )

        self.config = config or LongContextBenchmarkConfig()

        self.batch_size = batch_size

        self.include_mask = include_mask

        self.train_context_length = train_context_length

        self.generator = LongContextSampleGenerator(self.config)

        self._depths = self.generator._depths

        self._tasks = self.generator._tasks

        self._modes = self.generator._modes

        gate = set(self.config.primary_gate_task_types())
        self._retrieval_tasks = [t for t in self._tasks if t in gate]
        self._other_tasks = [t for t in self._tasks if t not in gate]
        self._overfit_pool: list[LongContextSample] | None = None
        n_overfit = int(self.config.overfit_train_samples)
        if n_overfit > 0:
            self._overfit_pool = self._build_overfit_pool(n_overfit)

    def _build_overfit_pool(self, count: int) -> list[LongContextSample]:
        rng = random.Random(self.config.seed)
        depths = self._depths
        modes = self._modes
        gen = self.generator.generate_one
        seq_len = self.train_context_length
        samples: list[LongContextSample] = []
        for i in range(count):
            samples.append(
                gen(
                    context_length=seq_len,
                    needle_depth=depths[i % len(depths)],
                    task_type=self._tasks[i % len(self._tasks)],
                    haystack_mode=modes[i % len(modes)],
                    seed=self.config.seed + i,
                )
            )
        return samples

    def _next_task_type(self, rng: random.Random, step_idx: int) -> str:
        sampling = self.config.train_task_sampling
        if sampling == "balanced":
            return self._tasks[step_idx % len(self._tasks)]
        if sampling == "retrieval_heavy" and self._retrieval_tasks:
            if rng.random() < 0.7:
                return rng.choice(self._retrieval_tasks)
            pool = self._other_tasks or self._tasks
            return rng.choice(pool)
        return rng.choice(self._tasks)



    def __iter__(self) -> Iterator[dict[str, torch.Tensor | None]]:
        if self._overfit_pool:
            yield from self._iter_overfit()
            return

        episode_batches = int(getattr(self.config, "placement_episode_batches", 0) or 0)
        if episode_batches > 1:
            yield from self._iter_placement_episodes(episode_batches)
            return

        worker = torch.utils.data.get_worker_info()

        stream_seed = self.config.seed + (worker.id if worker is not None else 0)

        rng = random.Random(stream_seed)

        seq_len = self.train_context_length

        gen = self.generator.generate_one

        bs = self.batch_size

        depths, modes = self._depths, self._modes

        step_idx = 0

        while True:

            samples = [

                gen(

                    context_length=seq_len,

                    needle_depth=rng.choice(depths),

                    task_type=self._next_task_type(rng, step_idx + i),

                    haystack_mode=rng.choice(modes),

                    seed=rng.randint(0, 2**31 - 1),

                )

                for i in range(bs)

            ]

            step_idx += bs

            out = _stack_samples(samples)

            if not self.include_mask:

                out.pop("attention_mask", None)

            yield out

    def _iter_placement_episodes(self, episode_batches: int) -> Iterator[dict[str, torch.Tensor | None]]:
        """Repeat fixed needles + query for ``episode_batches`` batches; only scatter changes."""
        worker = torch.utils.data.get_worker_info()
        stream_seed = self.config.seed + (worker.id if worker is not None else 0)
        rng = random.Random(stream_seed)
        seq_len = self.train_context_length
        gen = self.generator
        bs = self.batch_size
        depths, modes = self._depths, self._modes
        step_idx = 0
        episode_id = 0
        batches_left = 0
        episode_task: TaskPayload | None = None
        episode_task_type = ""
        episode_mode = ""
        episode_seed = 0

        while True:
            if batches_left <= 0:
                episode_task_type = self._next_task_type(rng, step_idx)
                episode_mode = rng.choice(modes)
                episode_seed = rng.randint(0, 2**31 - 1)
                episode_task, episode_mode = gen.generate_task_payload(
                    task_type=episode_task_type,
                    haystack_mode=episode_mode,
                    seed=episode_seed,
                )
                batches_left = episode_batches
                episode_id += 1

            batch_index = episode_batches - batches_left
            samples = []
            for i in range(bs):
                layout_seed = episode_seed + episode_id * 1_000_003 + batch_index * 10_007 + i * 1_009
                samples.append(
                    gen.assemble_from_task(
                        task=episode_task,
                        task_type=episode_task_type,
                        context_length=seq_len,
                        needle_depth=rng.choice(depths),
                        haystack_mode=episode_mode,
                        seed=layout_seed,
                    )
                )
            batches_left -= 1
            step_idx += bs
            out = _stack_samples(samples)
            if not self.include_mask:
                out.pop("attention_mask", None)
            yield out

    def _iter_overfit(self) -> Iterator[dict[str, torch.Tensor | None]]:
        pool = self._overfit_pool or []
        if not pool:
            return
        bs = self.batch_size
        idx = 0
        while True:
            batch_samples = [pool[(idx + i) % len(pool)] for i in range(bs)]
            idx += bs
            out = _stack_samples(batch_samples)
            if not self.include_mask:
                out.pop("attention_mask", None)
            yield out





class _PrefetchIterator:

    """Background CPU batch generation — overlaps with GPU forward/backward."""



    def __init__(self, source: Iterator, prefetch: int = 2):

        self._source = source

        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))

        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._thread.start()



    def _worker(self) -> None:

        try:

            for item in self._source:

                if self._stop.is_set():

                    break

                self._queue.put(item)

        finally:

            self._queue.put(None)



    def __iter__(self) -> Iterator:

        return self



    def __next__(self):

        item = self._queue.get()

        if item is None:

            self._stop.set()

            raise StopIteration

        return item



    def close(self) -> None:

        self._stop.set()





class LongContextEvalDataset(IterableDataset):

    """Fixed held-out eval grid (longest-first); batches pre-built on CPU."""



    def __init__(

        self,

        config: LongContextBenchmarkConfig | None = None,

        *,

        samples: list[LongContextSample] | None = None,

    ):

        self.config = config or LongContextBenchmarkConfig()

        if samples is not None:

            self._samples = samples

        else:

            self._samples = get_holdout_grid(self.config)

        self._batches = [_sample_to_eval_batch(s) for s in self._samples]



    def __iter__(self) -> Iterator[dict]:

        yield from self._batches



    def __len__(self) -> int:

        return len(self._batches)





def get_long_context_dataloader(

    config: LongContextBenchmarkConfig,

    *,

    split: str = "train",

    batch_size: int = 1,

    num_workers: int = 0,

    pin_memory: bool = False,

    prefetch_batches: int = 0,

    prefetch_factor: int = 4,

    holdout_samples: list[LongContextSample] | None = None,

    train_context_length: int | None = None,

) -> DataLoader | _PrefetchIterator:

    if split == "train":

        if train_context_length is None:

            raise ValueError("train_context_length required for train split")

        dataset: IterableDataset = LongContextTrainDataset(

            config,

            batch_size=batch_size,

            include_mask=False,

            train_context_length=train_context_length,

        )

        loader: DataLoader | _PrefetchIterator = DataLoader(

            dataset,

            batch_size=None,

            num_workers=num_workers,

            persistent_workers=num_workers > 0,

            prefetch_factor=prefetch_factor if num_workers > 0 else None,

        )

        if prefetch_batches > 0 and num_workers == 0:

            return _PrefetchIterator(iter(loader), prefetch=prefetch_batches)

        return loader

    dataset = LongContextEvalDataset(config, samples=holdout_samples)

    return DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)





def transfer_batch_to_device(

    batch: dict,

    device: torch.device,

    *,

    non_blocking: bool = True,

    pin_memory: bool = False,

) -> dict:

    """H2D transfer; CPU pin_memory in main process only (safe with num_workers>0)."""

    out = {}

    for key, val in batch.items():

        if key == "meta" or val is None:

            out[key] = val

        elif isinstance(val, torch.Tensor):

            tensor = val

            if pin_memory and device.type == "cuda" and not tensor.is_cuda and not tensor.is_pinned():

                tensor = tensor.pin_memory()

            out[key] = tensor.to(device, non_blocking=non_blocking and device.type == "cuda")

        else:

            out[key] = val

    return out


