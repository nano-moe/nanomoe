"""Run MuP-style coordinate checks for nanomoe models.

Example:
    uv run python -m nanomoe.experiments.coord_check --model_preset=tiny --steps=2 --seq_len=64
"""

from __future__ import annotations

from pathlib import Path

import chz
import torch

from nanomoe.model import MoEConfig
from nanomoe.model.coord_check import run_coord_check, save_coord_check_artifacts


@chz.chz
class CoordCheckConfig:
    model_preset: str = "tiny"
    width_multipliers: str = "1,2,4"
    depth_multipliers: str = "1,2,4"

    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    attention_type: str | None = "fsdp_attention"
    moe_kernel: str | None = None
    depth_alpha: float | None = None

    batch_size: int = 4
    seq_len: int = 128
    steps: int = 3
    num_seeds: int = 2
    seed: int = 0

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0

    dtype: str = "float32"
    device: str = "auto"

    output_dir: str = "checkpoints/coord_check"
    write_charts: bool = True


def _parse_multiplier_list(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"All multipliers must be >= 1, got {value} from '{spec}'")
        values.append(value)
    if not values:
        raise ValueError("At least one multiplier is required")
    return values


def main(cfg: CoordCheckConfig) -> None:
    base_config = getattr(MoEConfig, cfg.model_preset)()
    if cfg.num_experts is not None:
        base_config.num_experts = cfg.num_experts
    if cfg.num_experts_per_tok is not None:
        base_config.num_experts_per_tok = cfg.num_experts_per_tok
    if cfg.attention_type is not None:
        base_config.attention_type = cfg.attention_type
    if cfg.moe_kernel is not None:
        base_config.moe_kernel = cfg.moe_kernel
    if cfg.depth_alpha is not None:
        base_config.depth_alpha = cfg.depth_alpha

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if cfg.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Use one of {sorted(dtype_map)}")

    if cfg.device == "cpu" and cfg.dtype == "float16":
        raise ValueError("float16 coordinate checks on CPU are not supported; use float32 or bfloat16 instead.")

    width_multipliers = _parse_multiplier_list(cfg.width_multipliers)
    depth_multipliers = _parse_multiplier_list(cfg.depth_multipliers)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    print(f"Device: {device}, dtype: {cfg.dtype}")
    print(f"Base model preset: {cfg.model_preset}")
    print(f"Width multipliers: {width_multipliers}")
    print(f"Depth multipliers: {depth_multipliers}")
    print(f"Base hidden_size={base_config.hidden_size}, num_layers={base_config.num_layers}")
    print(f"Base num_experts={base_config.num_experts}, num_experts_per_tok={base_config.num_experts_per_tok}")

    result = run_coord_check(
        base_config,
        width_multipliers=width_multipliers,
        depth_multipliers=depth_multipliers,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        steps=cfg.steps,
        num_seeds=cfg.num_seeds,
        seed=cfg.seed,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
        device=device,
        dtype=dtype_map[cfg.dtype],
    )
    paths = save_coord_check_artifacts(
        result,
        Path(cfg.output_dir),
        write_charts=cfg.write_charts,
    )

    print(f"Wrote {len(result.records)} raw records and {len(result.summary)} summary records.")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    config = chz.entrypoint(CoordCheckConfig, allow_hyphens=True)
    main(config)
