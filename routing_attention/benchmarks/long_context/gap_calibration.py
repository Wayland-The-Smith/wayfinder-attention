"""Shared helpers for dense vs linear gap calibration (Phases 1–3)."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE = ROOT / "configs" / "routing_gap_screen_template.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "gap_calibration_2L_10k"

GAP_STOP_MIN = 0.25
DENSE_STOP_MIN = 0.60
DENSE_REJECT_MIN = 0.50
ADVANCE_GAP_MAX = 0.15


@dataclass
class KnobSettings:
    """One difficulty setting for gap calibration."""

    label: str
    scatter_multi_needles: bool | None = None
    num_distractors: int | None = None
    synthetic_decoy_addrs: int | None = None
    needle_depths: list[float] | None = None
    task_types: list[str] | None = None
    synthetic_conflict_rows: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label}
        if self.scatter_multi_needles is not None:
            out["scatter_multi_needles"] = self.scatter_multi_needles
        if self.num_distractors is not None:
            out["num_distractors"] = self.num_distractors
            out["synthetic_decoy_addrs"] = self.num_distractors
        if self.synthetic_decoy_addrs is not None:
            out["synthetic_decoy_addrs"] = self.synthetic_decoy_addrs
        if self.needle_depths is not None:
            out["needle_depths"] = self.needle_depths
        if self.task_types is not None:
            out["task_types"] = self.task_types
        if self.synthetic_conflict_rows is not None:
            out["synthetic_conflict_rows"] = self.synthetic_conflict_rows
        out.update(self.extra)
        return out


def load_template(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_TEMPLATE
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return raw.get("routing_arena", raw)


def build_arena_config(
    knobs: KnobSettings,
    *,
    train_t: int = 512,
    steps: int = 3000,
    validate_every: int = 500,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = copy.deepcopy(template or load_template())
    base["train_context_length"] = train_t
    base["description"] = f"gap calibration — {knobs.label} @ T={train_t}"
    bench = dict(base.get("long_context_benchmark", {}))
    bench["context_lengths"] = [train_t]
    for key, value in knobs.to_dict().items():
        if key == "label":
            continue
        bench[key] = value
    base["long_context_benchmark"] = bench
    tx = dict(base.get("transformer", {}))
    tx["sparse_finetune_steps"] = int(steps)
    tx["validate_every"] = int(validate_every)
    tx["validate_every_min"] = int(validate_every)
    base["transformer"] = tx
    base["dense_train_from_scratch"] = True
    base["dense_finetune_on_task"] = True
    base["dense_gate_min"] = 0
    return base


def write_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"routing_arena": cfg}, sort_keys=False), encoding="utf-8")


def extract_accuracy(payload: dict[str, Any], *, mid: bool = True) -> float | None:
    if not payload or payload.get("status") == "error":
        return None
    if mid:
        train_info = payload.get("train_info") or {}
        best = train_info.get("best_holdout") or {}
        acc = best.get("overall_accuracy")
        if acc is None:
            acc = best.get("primary_gate_accuracy")
        if acc is None and train_info.get("mid_validations"):
            last = train_info["mid_validations"][-1]
            acc = last.get("overall_accuracy") or last.get("primary_gate_accuracy")
        return float(acc) if acc is not None else None
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy")
    if acc is None:
        acc = ev.get("overall_accuracy")
    return float(acc) if acc is not None else None


def score_pair(dense_acc: float | None, linear_acc: float | None) -> float | None:
    if dense_acc is None or linear_acc is None:
        return None
    if dense_acc < DENSE_REJECT_MIN:
        return None
    if linear_acc > dense_acc:
        return None
    return (dense_acc - linear_acc) + 0.01 * dense_acc


def gap_met(dense_acc: float | None, linear_acc: float | None) -> bool:
    if dense_acc is None or linear_acc is None:
        return False
    return (dense_acc - linear_acc) >= GAP_STOP_MIN and dense_acc >= DENSE_STOP_MIN


def parse_run_results(latest_path: Path) -> dict[str, float | None]:
    if not latest_path.exists():
        return {"dense_flash": None, "linear": None}
    data = json.loads(latest_path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    out: dict[str, float | None] = {}
    for variant in ("dense_flash", "linear", "local_window64"):
        out[variant] = extract_accuracy(results.get(variant) or {}, mid=True)
    return out


def parse_run_results_official(latest_path: Path) -> dict[str, float | None]:
    if not latest_path.exists():
        return {"dense_flash": None, "linear": None, "local_window64": None}
    data = json.loads(latest_path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    out: dict[str, float | None] = {}
    for variant in ("dense_flash", "linear", "local_window64"):
        out[variant] = extract_accuracy(results.get(variant) or {}, mid=False)
    return out


def phase1_ladder() -> list[tuple[str, KnobSettings]]:
    return [
        ("L0", KnobSettings("baseline")),
        ("L1", KnobSettings("scatter", scatter_multi_needles=True)),
        ("L2_2", KnobSettings("decoys_2", num_distractors=2)),
        ("L2_4", KnobSettings("decoys_4", num_distractors=4)),
        ("L2_8", KnobSettings("decoys_8", num_distractors=8)),
        ("L3", KnobSettings("depth_0.90", needle_depths=[0.90])),
        (
            "L4",
            KnobSettings(
                "conflict_rows_3",
                task_types=["addr_val_conflict"],
                synthetic_conflict_rows=3,
                scatter_multi_needles=False,
            ),
        ),
    ]


CSV_FIELDS = [
    "phase",
    "run_id",
    "label",
    "knobs_json",
    "steps",
    "eval_mode",
    "dense_acc",
    "linear_acc",
    "local_acc",
    "gap",
    "score",
    "exit_code",
    "run_dir",
]


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDS})
