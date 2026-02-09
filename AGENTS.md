# AGENTS.md

This file orients coding agents to the nanomoe repo.

## Project overview
- nanomoe is a minimal Mixture-of-Experts (MoE) training stack with packed datasets and sampling utilities.
- Core package lives in `src/nanomoe/` (model, data, train, sample).
- Primary training entrypoint: `python -m nanomoe.experiments.pretrain`.
- Auxiliary scripts live in `scripts/` (SFT, long-running pretrain, prefetch benchmarks).

## Environment and setup
- Python >= 3.13.
- Use `uv` for deps and running code (preferred by existing scripts).

Common setup:
- `uv sync --dev`

Smoke run:
- `uv run python -m nanomoe.experiments.pretrain --max_steps=2 --max_examples=32`

## Tests
- Full test suite: `uv run pytest`
- MoE perf tests (opt-in): `NANOMOE_RUN_PERF=1 uv run pytest -s tests/test_moe.py`
- CUDA-only tests skip automatically if CUDA is unavailable.

## Lint / format / type check
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run ty check src`

## Training scripts
- Pretrain (full run): `scripts/pretrain_nemomath_full.sh`
- Pretrain (DDP): `scripts/pretrain_nemomath_full_ddp.sh`
- SFT example: `uv run python scripts/sft.py`
- Prefetch benchmarks: `scripts/bench_prefetch_on.sh`, `scripts/bench_prefetch_off.sh`

## Notes
- First runs download HuggingFace datasets/models; use `--max_examples` and low `--max_steps` for quick checks.
- Checkpoints/logs default to `checkpoints/` and write `metrics.jsonl` for perf scripts.
