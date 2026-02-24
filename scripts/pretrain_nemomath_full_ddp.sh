#!/usr/bin/env bash
set -euo pipefail

# Long-running pretraining on the Nemotron-CC-Math corpus with DDP.

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/pretrain_nemomath_full}
MODEL_PRESET=${MODEL_PRESET:-small}
NUM_EXPERTS=${NUM_EXPERTS:-}
NUM_EXPERTS_PER_TOK=${NUM_EXPERTS_PER_TOK:-}

DATASET_NAME=${DATASET_NAME:-nvidia/Nemotron-CC-Math-v1}
DATASET_CONFIG=${DATASET_CONFIG:-4plus}
TOKENIZER_NAME=${TOKENIZER_NAME:-Qwen/Qwen3-0.6B}
TEXT_KEY=${TEXT_KEY:-text}
SEQ_LEN=${SEQ_LEN:-8192}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-}
MIN_DOC_LEN=${MIN_DOC_LEN:-64}
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-10000}
MAX_EXAMPLES=${MAX_EXAMPLES:-}

BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
MAX_STEPS=${MAX_STEPS:-200000}
MAX_TOKENS=${MAX_TOKENS:-}

PEAK_LR=${PEAK_LR:-1e-4}
FLOOR_LR=${FLOOR_LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

LOG_EVERY=${LOG_EVERY:-20}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
WANDB_PROJECT=${WANDB_PROJECT:-nanomoe-pretrain}
WANDB_NAME=${WANDB_NAME:-}

DTYPE=${DTYPE:-bfloat16}
COMPILE_MODEL=${COMPILE_MODEL:-false}

DEVICE_PREFETCH=${DEVICE_PREFETCH:-true}
PREFETCH_BATCHES=${PREFETCH_BATCHES:-4}
PREFETCH_PIN_MEMORY=${PREFETCH_PIN_MEMORY:-true}
PREFETCH_NON_BLOCKING=${PREFETCH_NON_BLOCKING:-true}

PROFILE_CUDA=${PROFILE_CUDA:-false}
PROFILE_START_STEP=${PROFILE_START_STEP:-1}
PROFILE_STEPS=${PROFILE_STEPS:-5}

ARGS=(
  --distributed=true
  --model_preset="${MODEL_PRESET}"
  --dataset_name="${DATASET_NAME}"
  --dataset_config="${DATASET_CONFIG}"
  --tokenizer_name="${TOKENIZER_NAME}"
  --text_key="${TEXT_KEY}"
  --seq_len="${SEQ_LEN}"
  --min_doc_len="${MIN_DOC_LEN}"
  --shuffle_buffer="${SHUFFLE_BUFFER}"
  --batch_size="${BATCH_SIZE}"
  --gradient_accumulation="${GRADIENT_ACCUMULATION}"
  --max_steps="${MAX_STEPS}"
  --peak_lr="${PEAK_LR}"
  --floor_lr="${FLOOR_LR}"
  --warmup_steps="${WARMUP_STEPS}"
  --weight_decay="${WEIGHT_DECAY}"
  --max_grad_norm="${MAX_GRAD_NORM}"
  --log_every="${LOG_EVERY}"
  --checkpoint_every="${CHECKPOINT_EVERY}"
  --checkpoint_dir="${CHECKPOINT_DIR}"
  --wandb_project="${WANDB_PROJECT}"
  --dtype="${DTYPE}"
  --compile_model="${COMPILE_MODEL}"
  --device_prefetch="${DEVICE_PREFETCH}"
  --prefetch_batches="${PREFETCH_BATCHES}"
  --prefetch_pin_memory="${PREFETCH_PIN_MEMORY}"
  --prefetch_non_blocking="${PREFETCH_NON_BLOCKING}"
  --profile_cuda="${PROFILE_CUDA}"
  --profile_start_step="${PROFILE_START_STEP}"
  --profile_steps="${PROFILE_STEPS}"
)

if [[ -n "${NUM_EXPERTS}" ]]; then
  ARGS+=(--num_experts="${NUM_EXPERTS}")
fi
if [[ -n "${NUM_EXPERTS_PER_TOK}" ]]; then
  ARGS+=(--num_experts_per_tok="${NUM_EXPERTS_PER_TOK}")
fi
if [[ -n "${MAX_SEQ_LEN}" ]]; then
  ARGS+=(--max_seq_len="${MAX_SEQ_LEN}")
fi
if [[ -n "${MAX_EXAMPLES}" ]]; then
  ARGS+=(--max_examples="${MAX_EXAMPLES}")
fi
if [[ -n "${MAX_TOKENS}" ]]; then
  ARGS+=(--max_tokens="${MAX_TOKENS}")
fi
if [[ -n "${WANDB_NAME}" ]]; then
  ARGS+=(--wandb_name="${WANDB_NAME}")
fi

uv run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m nanomoe.experiments.pretrain "${ARGS[@]}"
