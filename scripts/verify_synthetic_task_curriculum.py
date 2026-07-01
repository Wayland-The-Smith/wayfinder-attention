"""Quick check for synthetic_task_curriculum and hop_count=1 ptr_chain."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_7 import _curriculum_synthetic_task_settings
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    generate_ptr_chain_hop_first,
)

CURRICULUM = [
    {"until_step": 5000, "synthetic_hop_count": 1, "num_distractors": 0},
    {"until_step": 10000, "synthetic_hop_count": 1, "num_distractors": 1},
    {"until_step": 30000, "synthetic_hop_count": 2, "num_distractors": 1},
    {"until_step": 50000, "synthetic_hop_count": 2, "num_distractors": 2},
]


def main() -> None:
    for step in [0, 4999, 5000, 9999, 10000, 29999, 30000, 49999]:
        print(step, _curriculum_synthetic_task_settings(step, CURRICULUM))

    rng = random.Random(0)
    for hops, decoys in [(1, 0), (1, 1), (2, 1), (2, 2)]:
        payload = generate_ptr_chain_hop_first(
            rng, hop_count=hops, num_distractors=decoys
        )
        ptr_hops = payload.metadata["ptr_hops"]
        print(
            f"hops={hops} decoys={decoys} ptr_hops={ptr_hops} "
            f"segs={len(payload.needle_segments)} ok"
        )
    print("verification ok")


if __name__ == "__main__":
    main()
