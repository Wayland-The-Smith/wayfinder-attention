#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fla311
cd /mnt/c/Users/anike/Desktop/Experiments/RoutingAttention/RoutingAttention

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python run_experiment_6_fused_suite.py "$@"
