# AGENTS.md

This file orients coding agents to the `nanomoe` repo.

## Project overview
- `nanomoe` is a minimal transformer/MoE training stack with packed-data utilities, sampling, and lightweight experiment entrypoints.
- Core package lives in `src/nanomoe/`.
- Existing pretraining entrypoint: `python -m nanomoe.experiments.pretrain`.
- Current LRA entrypoint: `python -m nanomoe.experiments.lra`.

## Environment and setup
- Python >= 3.13.
- Use `uv` for dependency management and execution.
- `torch>=2.10` is expected for modern attention paths and `F.grouped_mm`.

Common setup:
- `uv sync --dev`

Quick smoke runs:
- `uv run python -m nanomoe.experiments.pretrain --max_steps=2 --max_examples=32`
- `uv run python -m nanomoe.experiments.lra --task=listops --max-steps=2 --max-train-examples=64 --max-eval-examples=64`

## Repo layout
- `src/nanomoe/model/`: decoder-only LM stack, attention, MoE, model config.
- `src/nanomoe/data/`: packed streaming datasets and collators.
- `src/nanomoe/train/`: train loop, schedulers, checkpointing, logging.
- `src/nanomoe/experiments/`: runnable experiment entrypoints.
- `src/nanomoe/lra/`: LRA-specific data/model code for classification tasks.
- `tests/`: targeted unit tests.

## Tests
- Full suite: `uv run pytest`
- Focused LRA tests: `uv run pytest tests/test_lra_data.py tests/test_lra_model.py`
- CUDA-only tests skip automatically if CUDA is unavailable.

## Lint / format / type check
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run ty check src`
- `uv run pre-commit run --all-files`

## External reference repos

### `s4`
Path:
- `/home/ubuntu/s4`

Use it for:
- LRA dataset semantics and historical experiment targets.

Most relevant files:
- `/home/ubuntu/s4/src/dataloaders/lra.py`
- `/home/ubuntu/s4/configs/experiment/lra/s4-listops.yaml`
- `/home/ubuntu/s4/configs/experiment/lra/s4-pathx.yaml`
- `/home/ubuntu/s4/configs/pipeline/pathx.yaml`

Guidance:
- Treat `s4` as a reference implementation, not the base to extend.
- Reuse task definitions and data expectations, but keep `nanomoe` on modern `uv` and current PyTorch.

### `tf2d`
Path:
- `/home/ubuntu/tf2d`

Use it for:
- 2D hull-attention ideas and kernel/cache code.

Most relevant files:
- `/home/ubuntu/tf2d/h2_pretrain/model.py`
- `/home/ubuntu/tf2d/h2_pretrain/hull_cache.py`
- `/home/ubuntu/tf2d/h2_pretrain/hull2d_cuda_v4.cu`

Guidance:
- The current code is centered on autoregressive decode/cache, not full-sequence LRA training.
- Do not assume it is a drop-in attention replacement for classification.
- In `nanomoe`, the reserved backend name is `hullattn`, not `tf2d`.

### `transformers`
Path:
- `/home/ubuntu/transformers`

Use it for:
- HF implementation patterns, especially Qwen-style architecture details and config conventions.

Most relevant Qwen3.5 files:
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `/home/ubuntu/transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py`

Important note:
- `modeling_qwen3_5.py` is generated. If editing `transformers` itself, edit `modular_qwen3_5.py`, not the generated file.

Qwen3.5 text-model details worth borrowing carefully:
- RMSNorm is one-centered: output is multiplied by `(1 + weight)`.
- Attention uses GQA with separate `num_attention_heads`, `num_key_value_heads`, and explicit `head_dim`.
- Q and K get per-head RMSNorm (`q_norm`, `k_norm`) before RoPE.
- Q projection is gated: `q_proj` produces both query states and a learned gate; attention output is multiplied by `sigmoid(gate)` before `o_proj`.
- MLP is SwiGLU-style: `down_proj(act(gate_proj(x)) * up_proj(x))`.
- The decoder layer is pre-norm residual.
- Qwen3.5 alternates `linear_attention` and `full_attention` by `layer_types`; `nanomoe` does not currently mirror that structure.

## Current LRA work
- Modern LRA baseline work lives under `src/nanomoe/lra/`.
- Supported tasks right now:
  - `listops`
  - `pathx`
- Baseline model:
  - encoder-only Transformer classifier
  - attention backend names: `sdpa` and reserved `hullattn`
- `hullattn` currently raises `NotImplementedError`; the real integration is planned work.

## Working rules for this repo
- Prefer extending `nanomoe` rather than importing code directly from sibling repos.
- Keep new experiment code small and explicit; avoid introducing Hydra/Lightning-style complexity.
- Use `chz` config classes for experiment/model configs in new code.
- Preserve existing `uv` workflows and project style.
- When pulling ideas from HF, keep only the architectural parts that fit the local training stack.

## Notes
- First runs may download models or datasets; use `--max-train-examples`, `--max-eval-examples`, and low `--max-steps` for fast checks.
- Checkpoints and logs default to `checkpoints/` and write `metrics.jsonl`.
