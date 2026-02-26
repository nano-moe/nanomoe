#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/bench_prefetch_off}
MAX_STEPS=${MAX_STEPS:-20}
SEQ_LEN=${SEQ_LEN:-1024}
PREFETCH_BATCHES=${PREFETCH_BATCHES:-2}
DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-2-raw-v1}
TOKENIZER_NAME=${TOKENIZER_NAME:-gpt2}
MAX_EXAMPLES=${MAX_EXAMPLES:-200}
LOG_EVERY=${LOG_EVERY:-1}
SKIP_STEPS=${SKIP_STEPS:-2}
DEVICE_PREFETCH=false
PROFILE_CUDA=${PROFILE_CUDA:-true}
PROFILE_START_STEP=${PROFILE_START_STEP:-1}
PROFILE_STEPS=${PROFILE_STEPS:-5}

uv run python -m nanomoe.experiments.pretrain \
  --max_steps="${MAX_STEPS}" \
  --log_every="${LOG_EVERY}" \
  --model_preset=tiny \
  --seq_len="${SEQ_LEN}" \
  --prefetch_batches="${PREFETCH_BATCHES}" \
  --dataset_name="${DATASET_NAME}" \
  --dataset_config="${DATASET_CONFIG}" \
  --tokenizer_name="${TOKENIZER_NAME}" \
  --max_examples="${MAX_EXAMPLES}" \
  --wandb_project= \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --device_prefetch="${DEVICE_PREFETCH}" \
  --profile_cuda="${PROFILE_CUDA}" \
  --profile_start_step="${PROFILE_START_STEP}" \
  --profile_steps="${PROFILE_STEPS}" \
  --checkpoint_every=1000000 \
  --gradient_accumulation=1

python - <<'PY'
import json
import os
import statistics

checkpoint_dir = os.environ["CHECKPOINT_DIR"]
skip_steps = int(os.environ.get("SKIP_STEPS", "0"))
path = os.path.join(checkpoint_dir, "metrics.jsonl")

elapsed = []
tokens_per_sec = []
steps = 0

try:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if "perf/elapsed_time" not in entry:
                continue
            steps += 1
            if steps <= skip_steps:
                continue
            elapsed.append(float(entry["perf/elapsed_time"]))
            if "perf/tokens_per_sec" in entry:
                tokens_per_sec.append(float(entry["perf/tokens_per_sec"]))
except FileNotFoundError:
    raise SystemExit(f"Missing metrics file: {path}")

if not elapsed:
    raise SystemExit("No timing entries found; check log_every and metrics.jsonl.")

batches_per_sec = [1.0 / t for t in elapsed if t > 0]
mean_bps = statistics.fmean(batches_per_sec)
mean_tps = statistics.fmean(tokens_per_sec) if tokens_per_sec else 0.0

print(f"prefetch=off | steps={len(elapsed)} | avg_batches_per_sec={mean_bps:.4f} | avg_tokens_per_sec={mean_tps:.2f}")
PY
