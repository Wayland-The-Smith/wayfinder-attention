"""Logging and metrics persistence utilities."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def setup_logging(run_dir: Path, name: str = "routing_attention") -> logging.Logger:
    """Configure file + console logging for an experiment run."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ch = logging.StreamHandler(stream)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class MetricsLogger:
    """Persist scalar metrics to JSON and TensorBoard."""

    def __init__(self, tensorboard_dir: Path, stats_dir: Path):
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        self.stats_dir = stats_dir
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self._history: dict[str, list[dict[str, Any]]] = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        if tag not in self._history:
            self._history[tag] = []
        self._history[tag].append({"step": step, "value": value})

    def log_dict(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def flush_history(self, filename: str = "metrics_history.json") -> None:
        path = self.stats_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def close(self) -> None:
        self.flush_history()
        self.writer.close()
