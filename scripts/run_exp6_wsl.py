#!/usr/bin/env python3
"""Run Experiment 6 fused suite with CUDA preflight."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            print(f"  {name}: {fn()}")
else:
    print("WARNING: not using GPU")
    sys.exit(1)

from run_experiment_6_fused_suite import main

if __name__ == "__main__":
    main()
