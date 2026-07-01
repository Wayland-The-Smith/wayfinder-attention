#!/usr/bin/env bash
# Full NIAH gap suite: pointer_unique @ T=4096 + mqar_scatter @ T=2048 (dense + linear).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/experiments/Experiment_7/niah_gap_full_run.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$ROOT"
exec "$HOME/miniforge3/envs/fla311/bin/python" -u run_niah_gap_calibration_suite.py --experiment all >>"$LOG" 2>&1
