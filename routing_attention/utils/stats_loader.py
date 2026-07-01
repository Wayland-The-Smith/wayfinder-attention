"""Load experiment stats from run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from routing_attention.utils.experiment import find_latest_run, get_experiments_root


def load_run_stats(
    experiment_name: str,
    run_id: str | None = None,
    stat_name: str = "summary",
) -> dict[str, Any]:
    """
    Load a stats JSON file from an experiment run.

    Args:
        experiment_name: e.g. 'Experiment_1'
        run_id: e.g. 'run_001' or None for latest
        stat_name: filename without .json (e.g. 'summary', 'recall_metrics')
    """
    if run_id:
        run_dir = get_experiments_root() / experiment_name / run_id
    else:
        latest = find_latest_run(experiment_name)
        if latest is None:
            raise FileNotFoundError(f"No runs for {experiment_name}")
        run_dir = latest

    path = run_dir / "stats" / f"{stat_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(experiment_name: str) -> list[Path]:
    """List all run directories for an experiment."""
    exp_dir = get_experiments_root() / experiment_name
    if not exp_dir.exists():
        return []
    return sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
