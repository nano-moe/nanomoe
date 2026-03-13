#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

GPU_SET="${GPU_SET:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/s4/data_full}"
LOG_DIR="${LOG_DIR:-checkpoints/lra_full_sdpa}"
WANDB_PROJECT="${WANDB_PROJECT:-nanomoe-lra}"
WANDB_NAME="${WANDB_NAME:-listops-sdpa-ddp}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-42}"

D_MODEL="${D_MODEL:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-8}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CUDA_VISIBLE_DEVICES="$GPU_SET" uv run torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  -m nanomoe.experiments.lra \
  --task=listops \
  --distributed=true \
  --data_root="$DATA_ROOT" \
  --attention_backend=sdpa \
  --seed="$SEED" \
  --d_model="$D_MODEL" \
  --num_heads="$NUM_HEADS" \
  --num_layers="$NUM_LAYERS" \
  --ffn_hidden_size="$FFN_HIDDEN_SIZE" \
  --batch_size="$BATCH_SIZE" \
  --eval_batch_size="$EVAL_BATCH_SIZE" \
  --max_steps="$MAX_STEPS" \
  --eval_every="$EVAL_EVERY" \
  --num_workers="$NUM_WORKERS" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_name="$WANDB_NAME" \
  --wandb_mode="$WANDB_MODE" \
  --log_dir="$LOG_DIR"
