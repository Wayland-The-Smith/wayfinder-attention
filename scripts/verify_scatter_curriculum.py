"""Verify scatter curriculum expansion and phase resolution for ptr_chain."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_7 import (
    _apply_scatter_curriculum_spec,
    _curriculum_needle_scatter_settings,
    _curriculum_synthetic_task_settings,
    _expand_needle_scatter_curriculum,
)


def main() -> None:
    cfg_path = Path("configs/routing_arena_ptr_chain_t2048_scatter_curriculum.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    arena = raw["routing_arena"]
    spec = arena["long_context_benchmark"]["needle_scatter_curriculum_spec"]

    expanded = _expand_needle_scatter_curriculum(spec, train_context_length=2048)
    print(f"tiers={len(expanded)} final_until={expanded[-1]['until_step']}")
    print("first", expanded[0])
    print("last", expanded[-1])

    config = {"long_context_benchmark": arena["long_context_benchmark"]}
    _apply_scatter_curriculum_spec(config, 2048)
    curriculum = config["long_context_benchmark"]["needle_scatter_curriculum"]
    assert len(curriculum) == len(expanded)

    for step in [0, 749, 750, 1499, 30749, 30750, 49999]:
        scatter = _curriculum_needle_scatter_settings(step, curriculum)
        task = _curriculum_synthetic_task_settings(
            step, arena["long_context_benchmark"]["synthetic_task_curriculum"]
        )
        print(f"step={step} scatter={scatter} task={task}")

    print("verification ok")


if __name__ == "__main__":
    main()
