"""Experiment directory management and run orchestration."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routing_attention.utils.config import save_config


def get_project_root() -> Path:
    """Return repository root (parent of routing_attention package)."""
    return Path(__file__).resolve().parents[2]


def get_experiments_root() -> Path:
    """Return Experiments/ output directory at repo root."""
    return get_project_root() / "Experiments"


def get_next_run_dir(experiment_name: str) -> Path:
    """
    Get next run directory: Experiments/Experiment_N/run_XXX.

    experiment_name should be like 'Experiment_1'.
    """
    exp_dir = get_experiments_root() / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    for child in exp_dir.iterdir():
        match = re.match(r"run_(\d+)$", child.name)
        if match:
            existing.append(int(match.group(1)))

    next_idx = max(existing, default=0) + 1
    run_dir = exp_dir / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run(experiment_name: str) -> Path | None:
    """Find the most recent run directory for an experiment."""
    exp_dir = get_experiments_root() / experiment_name
    if not exp_dir.exists():
        return None
    runs = sorted(
        [p for p in exp_dir.iterdir() if re.match(r"run_\d+$", p.name)],
        key=lambda p: int(re.search(r"run_(\d+)$", p.name).group(1)),
    )
    return runs[-1] if runs else None


class ExperimentRunner:
    """Manages experiment run directories, config snapshots, and artifact paths."""

    def __init__(
        self,
        experiment_name: str,
        config: dict[str, Any],
        dry_run: bool = False,
        run_dir: Path | None = None,
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.dry_run = dry_run
        self.run_dir = run_dir or get_next_run_dir(experiment_name)

        if dry_run:
            self.config = _apply_dry_run_overrides(self.config)

        self._checkpoint_dir = self.run_dir / "checkpoints"
        self._setup_directories()
        self._save_run_metadata()

    def _setup_directories(self) -> None:
        for sub in ("checkpoints", "tensorboard", "plots", "stats", "data_cache"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

    def _save_run_metadata(self) -> None:
        save_config(self.config, self.run_dir / "config.yaml")
        metadata = {
            "experiment_name": self.experiment_name,
            "dry_run": self.dry_run,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(self.run_dir),
        }
        with open(self.run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, path: Path | str) -> None:
        self._checkpoint_dir = Path(path)

    def reset_checkpoint_dir(self) -> None:
        """Restore checkpoint_dir to the run's default checkpoints folder."""
        self._checkpoint_dir = self.run_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def stats_dir(self) -> Path:
        return self.run_dir / "stats"

    @property
    def data_cache_dir(self) -> Path:
        return self.run_dir / "data_cache"

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def finalize(self, summary: dict[str, Any]) -> None:
        """Write final summary stats and mark run complete."""
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        summary["dry_run"] = self.dry_run
        with open(self.stats_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def _apply_dry_run_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Override training steps for dry-run mode."""
    import copy

    cfg = copy.deepcopy(config)
    dry_steps = cfg.get("dry_run", {}).get("max_steps", 40)

    for section in ("transformer", "router", "routing_attention"):
        if section in cfg:
            cfg[section]["max_steps"] = dry_steps
            cfg[section]["eval_every"] = 0
            cfg[section]["save_every"] = dry_steps

    if "validation" in cfg:
        cfg["validation"]["enabled"] = False
        cfg["validation"]["eval_every"] = 0

    if "data_collection" in cfg:
        train_batches = min(cfg["data_collection"].get("train_max_batches", 256), 5)
        holdout_batches = min(cfg["data_collection"].get("holdout_max_batches", 32), 2)
        cfg["data_collection"]["train_max_batches"] = train_batches
        cfg["data_collection"]["holdout_max_batches"] = holdout_batches
        cfg["data_collection"]["max_batches"] = train_batches
        if cfg["data_collection"].get("per_head", False):
            cfg["data_collection"]["train_max_batches"] = 1
            cfg["data_collection"]["holdout_max_batches"] = 1
            cfg["data_collection"]["max_batches"] = 1
            cfg["data_collection"]["batch_size"] = min(
                cfg["data_collection"].get("batch_size", cfg.get("data", {}).get("batch_size", 32)),
                4,
            )

    if "evaluation" in cfg:
        cfg["evaluation"]["max_batches"] = min(
            cfg["evaluation"].get("max_batches", 100), 3
        )
        if cfg["evaluation"].get("max_eval_samples", 0) > 0:
            cfg["evaluation"]["max_eval_samples"] = min(
                cfg["evaluation"]["max_eval_samples"], 4
            )
        # Keep max_eval_tokens at 0 so resolve_max_eval_tokens uses full MNIST (784) in dry-run.
        cfg["evaluation"]["recall_max_batches"] = min(
            cfg["evaluation"].get("recall_max_batches", 10), 3
        )
        cfg["evaluation"]["benchmark_runs"] = min(
            cfg["evaluation"].get("benchmark_runs", 10), 2
        )
        cfg["evaluation"]["benchmark_warmup"] = min(
            cfg["evaluation"].get("benchmark_warmup", 3), 1
        )
        cfg["evaluation"]["include_mean_rank"] = False

    if "validation" in cfg:
        cfg["validation"]["eval_max_samples"] = min(
            cfg["validation"].get("eval_max_samples", 8), 4
        )

    return cfg


def resolve_checkpoint_path(
    path_str: str | None,
    experiment_name: str | None = None,
    artifact_name: str = "best.pt",
) -> Path | None:
    """
    Resolve a checkpoint path from explicit path, 'latest', or experiment reference.

    Formats:
      - absolute/relative path
      - 'latest:Experiment_1' -> latest run's checkpoint
      - 'Experiment_1/run_001/checkpoints/best.pt'
    """
    if not path_str:
        return None

    root = get_project_root()
    if path_str.startswith("latest:"):
        exp = path_str.split(":", 1)[1]
        latest = find_latest_run(exp)
        if latest is None:
            raise FileNotFoundError(f"No runs found for {exp}")
        return latest / "checkpoints" / artifact_name

    path = Path(path_str)
    if not path.is_absolute():
        candidates = [
            get_experiments_root() / path_str,
            root / path_str,
        ]
        normalized = path_str.replace("\\", "/")
        if normalized.lower().startswith("experiments/"):
            stripped = normalized.split("/", 1)[1]
            candidates.append(get_experiments_root() / stripped)
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return path if path.exists() else None


def resolve_run_dir(path_str: str | None) -> Path | None:
    """Resolve an experiment run directory (e.g. Experiment_1/run_025)."""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute() and path.is_dir():
        return path
    norm = str(path_str).replace("\\", "/")
    if norm.lower().startswith("experiments/"):
        norm = norm.split("/", 1)[1]
    for base in (get_experiments_root(), get_project_root()):
        candidate = base / norm
        if candidate.is_dir():
            return candidate.resolve()
    return None


def find_transformer_checkpoint_in_run(run_dir: Path) -> Path | None:
    """Prefer final.pt, then latest step checkpoint, then best.pt."""
    ckpt_root = run_dir / "checkpoints" / "transformer"
    for name in ("final.pt", "step_010000.pt", "step_005000.pt", "best.pt"):
        candidate = ckpt_root / name
        if candidate.exists():
            return candidate
    if ckpt_root.is_dir():
        steps = sorted(ckpt_root.glob("step_*.pt"), reverse=True)
        if steps:
            return steps[0]
    return None
