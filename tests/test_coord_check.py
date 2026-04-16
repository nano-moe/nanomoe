from __future__ import annotations

from pathlib import Path

import torch

from nanomoe.model import MoEConfig
from nanomoe.model.coord_check import run_coord_check, save_coord_check_artifacts, scale_moe_config


def _base_config(**kwargs: object) -> MoEConfig:
    return MoEConfig(
        hidden_size=32,
        num_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_type="fsdp_attention",
        **kwargs,
    )


def test_scale_moe_config_width_scaling_preserves_head_dim() -> None:
    base = _base_config(shared_expert=True, shared_expert_intermediate_size=48)
    scaled = scale_moe_config(base, width_multiplier=3)

    assert scaled.hidden_size == 96
    assert scaled.intermediate_size == 192
    assert scaled.num_attention_heads == 12
    assert scaled.num_key_value_heads == 12
    assert scaled.head_dim == base.head_dim
    assert scaled.shared_expert_intermediate_size == 144
    assert scaled.num_layers == base.num_layers


def test_scale_moe_config_depth_scaling_only_changes_layers() -> None:
    base = _base_config(depth_alpha=0.5)
    scaled = scale_moe_config(base, depth_multiplier=4)

    assert scaled.num_layers == 8
    assert scaled.hidden_size == base.hidden_size
    assert scaled.intermediate_size == base.intermediate_size
    assert scaled.depth_alpha == 0.5


def test_run_coord_check_collects_width_and_depth_records() -> None:
    result = run_coord_check(
        _base_config(),
        width_multipliers=(1, 2),
        depth_multipliers=(1, 2),
        batch_size=2,
        seq_len=8,
        steps=1,
        num_seeds=1,
        seed=0,
        lr=1e-3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.records
    assert result.summary

    axes = {record.axis for record in result.records}
    families = {record.family for record in result.records}
    scales = {(record.axis, record.scale) for record in result.records}

    assert axes == {"width", "depth"}
    assert {"activation", "gradient", "update_ratio", "loss"} <= families
    assert {("width", 1), ("width", 2), ("depth", 1), ("depth", 2)} <= scales

    layer_records = [record for record in result.records if record.layer_position is not None]
    assert layer_records
    assert all(0.0 < float(record.layer_position) <= 1.0 for record in layer_records)


def test_save_coord_check_artifacts_writes_outputs(tmp_path: Path) -> None:
    result = run_coord_check(
        _base_config(),
        width_multipliers=(1,),
        depth_multipliers=(1,),
        batch_size=1,
        seq_len=8,
        steps=1,
        num_seeds=1,
        seed=0,
        lr=1e-3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    paths = save_coord_check_artifacts(result, tmp_path, write_charts=False)

    assert paths["raw"].exists()
    assert paths["summary"].exists()
    assert paths["csv"].exists()
