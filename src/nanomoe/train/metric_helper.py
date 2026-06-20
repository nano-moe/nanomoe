from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanomoe.data.packed_dataset import PackedPretrainStreamGroup
from nanomoe.monitors import hidden_state_cosine_similarities


def _format_hparam_value(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m").replace("+", "")


def build_run_dir(args: argparse.Namespace) -> Path:
    run_parts = []
    dataset = getattr(args, "dataset", None)
    if dataset is not None:
        run_parts.append(f"data-{_format_hparam_value(dataset)}")
    seed = getattr(args, "seed", None)
    if seed is not None:
        run_parts.append(f"seed-{_format_hparam_value(seed)}")
    run_parts.extend(
        [
            f"opt-{args.optimizer}",
            f"lr-{_format_hparam_value(args.learning_rate)}",
            f"wd-{_format_hparam_value(args.weight_decay)}",
            f"steps-{args.iterations}",
            f"ga-{args.grad_accum}",
            f"warmup-{args.warmup_steps}",
            f"hm-{args.hidden_metrics_every}",
            f"depth-{_format_hparam_value(args.use_depth_scaling)}",
        ]
    )
    run_name = "_".join(run_parts)
    return args.log_dir / run_name


def capture_hidden_metrics(model: torch.nn.Module, dataset: PackedPretrainStreamGroup, step: int) -> dict[str, Any]:
    result = hidden_state_cosine_similarities(
        model=model,
        dataset=iter(dataset),
        batch_size=1,
        max_batches=1,
    )
    return {
        "step": step,
        "intra_sequence_mean_off_diagonal_cosine": [
            item.mean_off_diagonal_cosine for item in result.intra_sequence
        ],
        "neighbouring_mean_token_cosine": [item.mean_token_cosine for item in result.neighbouring_layers],
        "neighbouring_mean_rms_distance": [item.mean_rms_distance for item in result.neighbouring_layers],
        "neighbouring_mean_relative_rms_change": [
            item.mean_relative_rms_change for item in result.neighbouring_layers
        ],
        "first_layer_reference_mean_token_cosine": [
            item.mean_token_cosine for item in result.first_layer_reference
        ],
        "first_layer_reference_mean_rms_distance": [
            item.mean_rms_distance for item in result.first_layer_reference
        ],
        "first_layer_reference_mean_relative_rms_change": [
            item.mean_relative_rms_change for item in result.first_layer_reference
        ],
        "router_usage_entropy": [item.entropy for item in result.router_usage_entropy],
    }


def _stack_metric(records: list[dict[str, Any]], key: str, dtype: np.dtype | type) -> np.ndarray:
    if not records:
        return np.empty((0,), dtype=dtype)
    return np.asarray([record[key] for record in records], dtype=dtype)


def save_metrics(
    run_dir: Path,
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    hidden_state_records: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "config.npy", config, allow_pickle=True)

    train_metrics = {
        "step": _stack_metric(train_records, "step", np.int64),
        "lr": _stack_metric(train_records, "lr", np.float64),
        "loss": _stack_metric(train_records, "loss", np.float64),
        "aux_loss": _stack_metric(train_records, "aux_loss", np.float64),
        "tokens": _stack_metric(train_records, "tokens", np.int64),
        "router_monitor": _stack_metric(train_records, "router_monitor", np.float64),
    }
    np.save(run_dir / "train_metrics.npy", train_metrics, allow_pickle=True)

    hidden_metrics = {
        "step": _stack_metric(hidden_state_records, "step", np.int64),
        "intra_sequence_mean_off_diagonal_cosine": _stack_metric(
            hidden_state_records, "intra_sequence_mean_off_diagonal_cosine", np.float64
        ),
        "neighbouring_mean_token_cosine": _stack_metric(
            hidden_state_records, "neighbouring_mean_token_cosine", np.float64
        ),
        "neighbouring_mean_rms_distance": _stack_metric(
            hidden_state_records, "neighbouring_mean_rms_distance", np.float64
        ),
        "neighbouring_mean_relative_rms_change": _stack_metric(
            hidden_state_records, "neighbouring_mean_relative_rms_change", np.float64
        ),
        "first_layer_reference_mean_token_cosine": _stack_metric(
            hidden_state_records, "first_layer_reference_mean_token_cosine", np.float64
        ),
        "first_layer_reference_mean_rms_distance": _stack_metric(
            hidden_state_records, "first_layer_reference_mean_rms_distance", np.float64
        ),
        "first_layer_reference_mean_relative_rms_change": _stack_metric(
            hidden_state_records, "first_layer_reference_mean_relative_rms_change", np.float64
        ),
        "router_usage_entropy": _stack_metric(hidden_state_records, "router_usage_entropy", np.float64),
    }
    np.save(run_dir / "hidden_state_metrics.npy", hidden_metrics, allow_pickle=True)
