#!/usr/bin/env bash
set -euo pipefail

# Example:
#   OPTIMIZER=muon LR=5e-5 ITERATIONS=1600 USE_DEPTH_SCALING=true scripts/simple_pretrain.sh

uv run python -m nanomoe.train.simple_pretrain \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --iterations 20 \
  --grad-accum 16 \
  --warmup-steps 100 \
  --hidden-metrics-every 100 \
  --log-dir /home/xwang457/work/nanomoe/pretrain_log \
  --use-depth-scaling
