#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fla311
cd /mnt/c/Users/anike/Desktop/Experiments/RoutingAttention/RoutingAttention

python -c "
import torch, triton
from routing_attention.kernels.causal_topk import causal_topk_available
print('GPU:', torch.cuda.get_device_name())
print('torch', torch.__version__, 'triton', triton.__version__, 'kernels', causal_topk_available())
"

python scripts/benchmark_causal_topk.py \
  --seq-lens 512 1024 2048 4096 8192 16384 \
  --dims 32 64 \
  --runs 10 \
  --warmup 3
