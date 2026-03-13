# LRA Hull Attention Plan

## Goal

Train and compare:
- a modern Transformer baseline on LRA
- a Transformer variant with attention backend swapped to `hullattn`

Primary implementation base:
- `/home/ubuntu/nanomoe`

Reference repos:
- `/home/ubuntu/s4`
- `/home/ubuntu/tf2d`
- `/home/ubuntu/transformers`

## Current status

Implemented in `nanomoe`:
- `ListOps` loader
- `Path-X` loader via Pathfinder-128 sequential grayscale inputs
- `Path-X` retrieval helper at `scripts/get_lra_data.sh`
- encoder-only Transformer classifier baseline
- `python -m nanomoe.experiments.lra`
- attention backend seam with names:
  - `sdpa`
  - `hullattn` (reference implementation)

Verified:
- `uv run pytest tests/test_lra_data.py tests/test_lra_model.py`
- `pathx` now loads both text-style metadata fixtures and real binary `.npy` metadata from mirrored Pathfinder-128 drops

Current limitation:
- `hullattn` now has a correctness-first reference implementation in `nanomoe`.
- Current `hullattn` semantics:
  - brute-force full-sequence top-k sparse attention
  - padding-mask support
  - `head_dim=2` only
- Current `hullattn` limitation:
  - no convex-hull acceleration yet
  - no decode/KV-cache integration
  - much slower than `sdpa` at long sequence lengths

## Why `nanomoe` is the base

Reasons:
- already uses `uv`
- already targets modern PyTorch
- already has lightweight training/checkpoint/logging infrastructure
- easier to evolve than retrofitting old `s4` Hydra/Lightning code

Why not use `s4` directly:
- older framework assumptions
- older dependency stack
- higher migration cost before baseline iteration

Why not use `tf2d` directly:
- current code is decode/cache oriented
- not a full-sequence LRA classification training stack

## External references

### `s4`

Use for task semantics and rough target configs.

Key files:
- `/home/ubuntu/s4/src/dataloaders/lra.py`
- `/home/ubuntu/s4/configs/experiment/lra/s4-listops.yaml`
- `/home/ubuntu/s4/configs/experiment/lra/s4-pathx.yaml`
- `/home/ubuntu/s4/configs/pipeline/pathx.yaml`

Relevant takeaways:
- `ListOps` uses token remapping consistent with the original LRA preprocessing.
- `Path-X` is represented through Pathfinder-128 sequential grayscale inputs.
- the old `lra_release.gz` URL used by older docs may no longer be reliable; keep a mirror fallback for `pathfinder128`

### `tf2d`

Use for hull-attention mechanics and possible CUDA reuse.

Key files:
- `/home/ubuntu/tf2d/h2_pretrain/model.py`
- `/home/ubuntu/tf2d/h2_pretrain/hull_cache.py`
- `/home/ubuntu/tf2d/h2_pretrain/hull2d_cuda_v4.cu`

Relevant takeaways:
- existing code is based around `head_dim=2`
- current implementation is best understood as a decode-time hull cache
- there is not yet a ready-made full-sequence training attention module for LRA classification

### `transformers`

Use for model-architecture reference quality, not as a direct dependency target.

Primary Qwen3.5 text references:
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py`

Relevant takeaways for `nanomoe`:
- one-centered RMSNorm
- GQA with explicit `head_dim`
- per-head `q_norm` and `k_norm`
- gated Q projection in attention
- SwiGLU MLP
- pre-norm residual block structure

## Proposed phases

### Phase 1: Baseline stabilization

Objective:
- make the `sdpa` baseline solid and runnable on both `ListOps` and `Path-X`

Tasks:
- run short real-data smoke jobs for both tasks
- verify training loss decreases
- verify eval/checkpoint/logging behavior
- tune obvious defaults for batch size, LR, depth, and pooling

Deliverables:
- known-good launch commands
- one baseline result per task

### Phase 2: Qwen-style cleanup pass

Objective:
- decide how much Qwen3.5 structure to import into the classifier baseline

Likely changes:
- switch current RMS/LayerNorm choices toward Qwen-style RMSNorm
- consider GQA support
- consider SwiGLU FFN instead of current simpler FFN
- decide whether attention gating is worth adding before `hullattn`

Constraint:
- keep this an encoder-classification stack, not a decoder LM port

Recommendation:
- borrow only the parts that simplify later `hullattn` work

### Phase 3: Define `hullattn` contract

Objective:
- specify exactly what `hullattn` means in `nanomoe`

Decisions required:
- full-sequence bidirectional attention or causal-only approximation
- exact argmax-style token selection vs weighted sparse attention
- whether `head_dim` must equal 2
- whether to require all heads use 2D keys or only a specialized path

Minimum viable contract:
- `hidden_states -> q, k, v`
- attention backend consumes `[B, H, T, D]`
- returns same output shape as SDPA
- supports padding mask
- no KV-cache requirement for LRA classification path

### Phase 4: First real `hullattn` implementation

Objective:
- replace placeholder backend with a real training/eval implementation

Possible approaches:
- pure PyTorch reference implementation first, CUDA later
- adapt `tf2d` concepts into a full-sequence module
- start with exact top-1/argmax-style token retrieval if weighted attention is too large a jump

Recommendation:
- build a slow but correct PyTorch reference path first
- only then decide whether to port or wrap CUDA from `tf2d`

Why:
- correctness and integration debugging are easier before kernel work

### Phase 5: Comparison runs

Objective:
- compare `sdpa` baseline, `hullattn`, and if useful `s4` historical baselines

Comparison set:
- `nanomoe` Transformer baseline on `ListOps`
- `nanomoe` Transformer baseline on `Path-X`
- `nanomoe` `hullattn` model on the same tasks
- optional `s4` reference numbers from configs or reruns

Metrics:
- validation accuracy
- test accuracy
- training throughput
- peak memory

## Immediate next steps

1. Run real-data smoke training for `ListOps` and `Path-X` with `sdpa`.
2. Run matching smoke jobs for `hullattn` with `head_dim=2`.
3. Measure throughput / memory cost versus `sdpa`.
4. Decide whether to keep the current classifier block or add a small Qwen-lite cleanup pass.
5. Replace brute-force top-k selection with a real hull-based implementation.

## Dataset bootstrap notes

Path-X source expected by `nanomoe`:
- `data/pathfinder/pathfinder128/curv_contour_length_14/...`
- or any alternate root passed via `--data-root` / `NANOMOE_LRA_DATA`

Bootstrap command:

```bash
./scripts/get_lra_data.sh
```

Behavior:
- tries the historical LRA archive first
- falls back to the public Pathfinder-128 mirror if that archive is unavailable
- leaves data in the layout already expected by `load_lra_datasets("pathx", ...)`

## Suggested commands

ListOps baseline smoke run:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=listops \
  --attention_backend=sdpa \
  --max_steps=20 \
  --max_train_examples=512 \
  --max_eval_examples=256
```

ListOps baseline 4-GPU DDP smoke run:

```bash
uv run torchrun --standalone --nproc_per_node=4 -m nanomoe.experiments.lra \
  --task=listops \
  --distributed=true \
  --attention_backend=sdpa \
  --max_steps=20 \
  --max_train_examples=512 \
  --max_eval_examples=256
```

Path-X baseline smoke run:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=pathx \
  --attention_backend=sdpa \
  --max_steps=20 \
  --max_train_examples=256 \
  --max_eval_examples=64
```

Path-X baseline 4-GPU DDP smoke run:

```bash
uv run torchrun --standalone --nproc_per_node=4 -m nanomoe.experiments.lra \
  --task=pathx \
  --distributed=true \
  --attention_backend=sdpa \
  --max_steps=20 \
  --max_train_examples=256 \
  --max_eval_examples=64
```

ListOps `hullattn` smoke run:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=listops \
  --attention_backend=hullattn \
  --d_model=128 \
  --num_heads=64 \
  --hull_top_k=8 \
  --max_steps=20 \
  --max_train_examples=512 \
  --max_eval_examples=256
```

ListOps `hullattn` 4-GPU DDP smoke run:

```bash
uv run torchrun --standalone --nproc_per_node=4 -m nanomoe.experiments.lra \
  --task=listops \
  --distributed=true \
  --attention_backend=hullattn \
  --d_model=128 \
  --num_heads=64 \
  --hull_top_k=8 \
  --max_steps=20 \
  --max_train_examples=512 \
  --max_eval_examples=256
```

Path-X `hullattn` smoke run:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=pathx \
  --attention_backend=hullattn \
  --d_model=128 \
  --num_heads=64 \
  --hull_top_k=8 \
  --max_steps=20 \
  --max_train_examples=256 \
  --max_eval_examples=64
```

Path-X `hullattn` 4-GPU DDP smoke run:

```bash
uv run torchrun --standalone --nproc_per_node=4 -m nanomoe.experiments.lra \
  --task=pathx \
  --distributed=true \
  --attention_backend=hullattn \
  --d_model=128 \
  --num_heads=64 \
  --hull_top_k=8 \
  --max_steps=20 \
  --max_train_examples=256 \
  --max_eval_examples=64
```

Longer baseline training run template:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=listops \
  --attention_backend=sdpa \
  --batch_size=32 \
  --eval_batch_size=64 \
  --max_steps=1000 \
  --eval_every=100 \
  --checkpoint_every=200
```

Longer `hullattn` training run template:

```bash
uv run python -m nanomoe.experiments.lra \
  --task=listops \
  --attention_backend=hullattn \
  --d_model=128 \
  --num_heads=64 \
  --hull_top_k=8 \
  --batch_size=16 \
  --eval_batch_size=32 \
  --max_steps=1000 \
  --eval_every=100 \
  --checkpoint_every=200
```

Notes:
- `hullattn` currently requires `head_dim=2`, so `num_heads` must equal `d_model / 2`.
- For fair baseline comparisons, keep `d_model` fixed and increase `num_heads` for `hullattn` until `head_dim=2`.
- That preserves total attention width because `num_heads * head_dim = d_model` in both variants.
- Example at `d_model=128`:
  - baseline `sdpa`: `num_heads=8`, `head_dim=16`
  - `hullattn`: `num_heads=64`, `head_dim=2`
- With the current brute-force reference path, `hullattn` will usually need smaller batch sizes than `sdpa`.
- For apples-to-apples comparison, keep all non-attention hyperparameters matched unless memory forces a change.

Focused tests:

```bash
uv run pytest tests/test_lra_data.py tests/test_lra_model.py
```
