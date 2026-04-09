"""MuP-style coordinate checks for MoE models.

The utilities in this module run short synthetic training traces while varying
model width or depth and record activation, gradient, parameter, and update
statistics in a long-table format that is easy to inspect or plot.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanomoe.model.config import MoEConfig
from nanomoe.model.model import MoETransformer, create_model


@dataclass(slots=True)
class CoordCheckRecord:
    axis: str
    scale: int
    seed: int
    step: int
    family: str
    probe: str
    stat: str
    value: float
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    layer_index: int | None = None
    layer_position: float | None = None


@dataclass(slots=True)
class CoordCheckSummaryRecord:
    axis: str
    scale: int
    step: int
    family: str
    probe: str
    stat: str
    value: float
    num_runs: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    layer_index: int | None = None
    layer_position: float | None = None


@dataclass(slots=True)
class CoordCheckResult:
    records: list[CoordCheckRecord]
    summary: list[CoordCheckSummaryRecord]


def scale_moe_config(
    base_config: MoEConfig,
    *,
    width_multiplier: int = 1,
    depth_multiplier: int = 1,
) -> MoEConfig:
    """Create a width- or depth-scaled MoE config.

    Width scaling keeps the attention head dimension fixed by scaling
    `hidden_size`, `intermediate_size`, and the attention head counts together.
    Depth scaling multiplies `num_layers` while leaving width fixed.
    """
    if width_multiplier < 1:
        raise ValueError(f"width_multiplier must be >= 1, got {width_multiplier}")
    if depth_multiplier < 1:
        raise ValueError(f"depth_multiplier must be >= 1, got {depth_multiplier}")

    base_head_dim = base_config.head_dim or (base_config.hidden_size // base_config.num_attention_heads)
    data = base_config.to_dict()

    data["hidden_size"] = base_config.hidden_size * width_multiplier
    data["intermediate_size"] = base_config.intermediate_size * width_multiplier
    data["num_attention_heads"] = base_config.num_attention_heads * width_multiplier
    data["num_key_value_heads"] = base_config.num_key_value_heads * width_multiplier
    data["head_dim"] = base_head_dim
    data["num_layers"] = base_config.num_layers * depth_multiplier

    shared_size = base_config.shared_expert_intermediate_size
    if shared_size is not None:
        data["shared_expert_intermediate_size"] = shared_size * width_multiplier

    return MoEConfig.from_dict(data)


def summarize_coord_check(records: Sequence[CoordCheckRecord]) -> list[CoordCheckSummaryRecord]:
    grouped: dict[tuple[Any, ...], tuple[float, int]] = {}
    for record in records:
        key = (
            record.axis,
            record.scale,
            record.step,
            record.family,
            record.probe,
            record.stat,
            record.num_layers,
            record.hidden_size,
            record.intermediate_size,
            record.num_attention_heads,
            record.num_key_value_heads,
            record.num_experts,
            record.num_experts_per_tok,
            record.layer_index,
            record.layer_position,
        )
        total, count = grouped.get(key, (0.0, 0))
        grouped[key] = (total + record.value, count + 1)

    summary: list[CoordCheckSummaryRecord] = []
    for key, (total, count) in sorted(grouped.items()):
        (
            axis,
            scale,
            step,
            family,
            probe,
            stat,
            num_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            num_experts,
            num_experts_per_tok,
            layer_index,
            layer_position,
        ) = key
        summary.append(
            CoordCheckSummaryRecord(
                axis=axis,
                scale=scale,
                step=step,
                family=family,
                probe=probe,
                stat=stat,
                value=total / count,
                num_runs=count,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                layer_index=layer_index,
                layer_position=layer_position,
            )
        )
    return summary


def run_coord_check(
    base_config: MoEConfig,
    *,
    width_multipliers: Sequence[int] = (1, 2, 4),
    depth_multipliers: Sequence[int] = (1, 2, 4),
    batch_size: int = 4,
    seq_len: int = 128,
    steps: int = 3,
    num_seeds: int = 2,
    seed: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> CoordCheckResult:
    """Run MuP-style coordinate checks over width and depth sweeps."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2 for next-token loss, got {seq_len}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")

    resolved_device = _resolve_device(device)
    batches = _make_synthetic_batches(
        steps=steps,
        num_seeds=num_seeds,
        base_seed=seed,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=base_config.vocab_size,
        device=resolved_device,
    )

    records: list[CoordCheckRecord] = []

    for axis, multipliers in (("width", width_multipliers), ("depth", depth_multipliers)):
        for scale in multipliers:
            for seed_idx in range(num_seeds):
                run_seed = seed + seed_idx
                torch.manual_seed(run_seed)
                if resolved_device.type == "cuda":
                    torch.cuda.manual_seed_all(run_seed)

                scaled_config = (
                    scale_moe_config(base_config, width_multiplier=scale)
                    if axis == "width"
                    else scale_moe_config(base_config, depth_multiplier=scale)
                )
                model = create_model(scaled_config).to(device=resolved_device, dtype=dtype)
                _unwrap_compiled_attention(model)
                model.train()
                optimizer = _create_optimizer(
                    optimizer_name,
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )

                collector = _ActivationCollector(records, axis=axis, scale=scale, seed=run_seed, config=scaled_config)
                handles = _register_activation_hooks(model, collector)
                try:
                    for step_idx, input_ids in enumerate(batches[seed_idx]):
                        optimizer.zero_grad(set_to_none=True)
                        collector.set_step(step_idx)

                        parameter_snapshots = {
                            probe_name: parameter.detach().clone()
                            for probe_name, parameter, _, _ in _iter_parameter_probes(model)
                        }

                        outputs = model(input_ids=input_ids, use_cache=False)
                        _append_tensor_stats(
                            records,
                            axis=axis,
                            scale=scale,
                            seed=run_seed,
                            step=step_idx,
                            family="activation",
                            probe="final_hidden",
                            tensor=outputs.hidden_states,
                            config=scaled_config,
                        )
                        _append_tensor_stats(
                            records,
                            axis=axis,
                            scale=scale,
                            seed=run_seed,
                            step=step_idx,
                            family="activation",
                            probe="logits",
                            tensor=outputs.logits,
                            config=scaled_config,
                        )

                        token_loss = F.cross_entropy(
                            outputs.logits[:, :-1, :].reshape(-1, scaled_config.vocab_size),
                            input_ids[:, 1:].reshape(-1),
                        )
                        aux_loss = outputs.aux_loss
                        if not isinstance(aux_loss, Tensor):
                            aux_loss = torch.tensor(aux_loss, device=resolved_device, dtype=token_loss.dtype)
                        total_loss = token_loss + aux_loss

                        _append_scalar(
                            records,
                            axis=axis,
                            scale=scale,
                            seed=run_seed,
                            step=step_idx,
                            family="loss",
                            probe="token_loss",
                            value=float(token_loss.detach().item()),
                            config=scaled_config,
                        )
                        _append_scalar(
                            records,
                            axis=axis,
                            scale=scale,
                            seed=run_seed,
                            step=step_idx,
                            family="loss",
                            probe="aux_loss",
                            value=float(aux_loss.detach().item()),
                            config=scaled_config,
                        )
                        _append_scalar(
                            records,
                            axis=axis,
                            scale=scale,
                            seed=run_seed,
                            step=step_idx,
                            family="loss",
                            probe="total_loss",
                            value=float(total_loss.detach().item()),
                            config=scaled_config,
                        )

                        total_loss.backward()

                        for probe_name, parameter, layer_index, layer_position in _iter_parameter_probes(model):
                            before = parameter_snapshots[probe_name]
                            _append_tensor_stats(
                                records,
                                axis=axis,
                                scale=scale,
                                seed=run_seed,
                                step=step_idx,
                                family="parameter",
                                probe=probe_name,
                                tensor=before,
                                config=scaled_config,
                                layer_index=layer_index,
                                layer_position=layer_position,
                            )
                            if parameter.grad is not None:
                                _append_tensor_stats(
                                    records,
                                    axis=axis,
                                    scale=scale,
                                    seed=run_seed,
                                    step=step_idx,
                                    family="gradient",
                                    probe=probe_name,
                                    tensor=parameter.grad,
                                    config=scaled_config,
                                    layer_index=layer_index,
                                    layer_position=layer_position,
                                )

                        optimizer.step()

                        for probe_name, parameter, layer_index, layer_position in _iter_parameter_probes(model):
                            before = parameter_snapshots[probe_name]
                            delta = parameter.detach() - before
                            _append_tensor_stats(
                                records,
                                axis=axis,
                                scale=scale,
                                seed=run_seed,
                                step=step_idx,
                                family="update",
                                probe=probe_name,
                                tensor=delta,
                                config=scaled_config,
                                layer_index=layer_index,
                                layer_position=layer_position,
                            )
                            _append_ratio_stats(
                                records,
                                axis=axis,
                                scale=scale,
                                seed=run_seed,
                                step=step_idx,
                                probe=probe_name,
                                numerator=delta,
                                denominator=before,
                                config=scaled_config,
                                layer_index=layer_index,
                                layer_position=layer_position,
                            )
                finally:
                    for handle in handles:
                        handle.remove()

    return CoordCheckResult(records=records, summary=summarize_coord_check(records))


def save_coord_check_artifacts(
    result: CoordCheckResult,
    output_dir: str | Path,
    *,
    write_charts: bool = True,
) -> dict[str, Path]:
    """Persist raw records, summary records, and optional HTML charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path = output_path / "coord_check_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    summary_path = output_path / "coord_check_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as handle:
        for record in result.summary:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    csv_path = output_path / "coord_check_summary.csv"
    _write_summary_csv(result.summary, csv_path)

    paths = {
        "raw": raw_path,
        "summary": summary_path,
        "csv": csv_path,
    }

    if write_charts:
        chart_paths = _write_summary_charts(result.summary, output_path)
        paths.update(chart_paths)

    return paths


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_synthetic_batches(
    *,
    steps: int,
    num_seeds: int,
    base_seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> list[list[Tensor]]:
    batches: list[list[Tensor]] = []
    for seed_idx in range(num_seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(base_seed + seed_idx)
        seed_batches: list[Tensor] = []
        for _ in range(steps):
            batch = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                generator=generator,
                device="cpu",
            )
            seed_batches.append(batch.to(device))
        batches.append(seed_batches)
    return batches


def _create_optimizer(
    optimizer_name: str,
    parameters: Iterable[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    normalized = optimizer_name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}. Use 'adamw' or 'sgd'.")


def _unwrap_compiled_attention(model: MoETransformer) -> None:
    for layer in model.layers:
        forward = layer.self_attn.forward
        wrapped = getattr(forward, "__wrapped__", None)
        if wrapped is not None:
            layer.self_attn.forward = wrapped.__get__(layer.self_attn, type(layer.self_attn))


def _iter_parameter_probes(
    model: MoETransformer,
) -> Iterable[tuple[str, nn.Parameter, int | None, float | None]]:
    yield "embed_tokens.weight", model.embed_tokens.weight, None, None

    num_layers = len(model.layers)
    for layer_idx, layer in enumerate(model.layers):
        layer_position = (layer_idx + 1) / num_layers
        prefix = f"layers.{layer_idx}"
        yield f"{prefix}.self_attn.q_proj.weight", layer.self_attn.q_proj.weight, layer_idx, layer_position
        yield f"{prefix}.self_attn.o_proj.weight", layer.self_attn.o_proj.weight, layer_idx, layer_position

        router = getattr(layer.mlp, "router", None)
        gate = getattr(router, "gate", None)
        if isinstance(gate, nn.Linear):
            yield f"{prefix}.mlp.router.gate.weight", gate.weight, layer_idx, layer_position

        experts = getattr(layer.mlp, "experts", None)
        if experts is not None:
            yield f"{prefix}.mlp.experts.gate_up_proj", experts.gate_up_proj, layer_idx, layer_position
            yield f"{prefix}.mlp.experts.down_proj", experts.down_proj, layer_idx, layer_position
        else:
            dense_ffn = getattr(layer.mlp, "ffn", None)
            if dense_ffn is not None:
                yield f"{prefix}.mlp.ffn.gate_proj.weight", dense_ffn.gate_proj.weight, layer_idx, layer_position
                yield f"{prefix}.mlp.ffn.down_proj.weight", dense_ffn.down_proj.weight, layer_idx, layer_position

    yield "norm.weight", model.norm.weight, None, None
    if model.lm_head is not None:
        yield "lm_head.weight", model.lm_head.weight, None, None


def _append_scalar(
    records: list[CoordCheckRecord],
    *,
    axis: str,
    scale: int,
    seed: int,
    step: int,
    family: str,
    probe: str,
    value: float,
    config: MoEConfig,
    layer_index: int | None = None,
    layer_position: float | None = None,
) -> None:
    records.append(
        CoordCheckRecord(
            axis=axis,
            scale=scale,
            seed=seed,
            step=step,
            family=family,
            probe=probe,
            stat="value",
            value=value,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            layer_index=layer_index,
            layer_position=layer_position,
        )
    )


def _append_tensor_stats(
    records: list[CoordCheckRecord],
    *,
    axis: str,
    scale: int,
    seed: int,
    step: int,
    family: str,
    probe: str,
    tensor: Tensor | None,
    config: MoEConfig,
    layer_index: int | None = None,
    layer_position: float | None = None,
) -> None:
    if tensor is None:
        return
    detached = tensor.detach().float()
    mean_abs = detached.abs().mean().item()
    rms = detached.pow(2).mean().sqrt().item()
    _append_scalar(
        records,
        axis=axis,
        scale=scale,
        seed=seed,
        step=step,
        family=family,
        probe=probe,
        value=mean_abs,
        config=config,
        layer_index=layer_index,
        layer_position=layer_position,
    )
    records[-1].stat = "mean_abs"
    _append_scalar(
        records,
        axis=axis,
        scale=scale,
        seed=seed,
        step=step,
        family=family,
        probe=probe,
        value=rms,
        config=config,
        layer_index=layer_index,
        layer_position=layer_position,
    )
    records[-1].stat = "rms"


def _append_ratio_stats(
    records: list[CoordCheckRecord],
    *,
    axis: str,
    scale: int,
    seed: int,
    step: int,
    probe: str,
    numerator: Tensor,
    denominator: Tensor,
    config: MoEConfig,
    layer_index: int | None = None,
    layer_position: float | None = None,
) -> None:
    eps = torch.finfo(torch.float32).eps
    numerator_f = numerator.detach().float()
    denominator_f = denominator.detach().float()

    mean_abs_ratio = numerator_f.abs().mean().item() / max(denominator_f.abs().mean().item(), eps)
    rms_ratio = numerator_f.pow(2).mean().sqrt().item() / max(denominator_f.pow(2).mean().sqrt().item(), eps)

    _append_scalar(
        records,
        axis=axis,
        scale=scale,
        seed=seed,
        step=step,
        family="update_ratio",
        probe=probe,
        value=mean_abs_ratio,
        config=config,
        layer_index=layer_index,
        layer_position=layer_position,
    )
    records[-1].stat = "mean_abs"
    _append_scalar(
        records,
        axis=axis,
        scale=scale,
        seed=seed,
        step=step,
        family="update_ratio",
        probe=probe,
        value=rms_ratio,
        config=config,
        layer_index=layer_index,
        layer_position=layer_position,
    )
    records[-1].stat = "rms"


def _extract_tensor(output: Any) -> Tensor | None:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, Tensor):
                return item
    return None


class _ActivationCollector:
    def __init__(
        self,
        records: list[CoordCheckRecord],
        *,
        axis: str,
        scale: int,
        seed: int,
        config: MoEConfig,
    ) -> None:
        self.records = records
        self.axis = axis
        self.scale = scale
        self.seed = seed
        self.config = config
        self.step = 0

    def set_step(self, step: int) -> None:
        self.step = step

    def make_hook(
        self,
        probe: str,
        *,
        layer_index: int | None = None,
        layer_position: float | None = None,
    ):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_tensor(output)
            _append_tensor_stats(
                self.records,
                axis=self.axis,
                scale=self.scale,
                seed=self.seed,
                step=self.step,
                family="activation",
                probe=probe,
                tensor=tensor,
                config=self.config,
                layer_index=layer_index,
                layer_position=layer_position,
            )

        return _hook


def _register_activation_hooks(
    model: MoETransformer,
    collector: _ActivationCollector,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    handles.append(model.embed_tokens.register_forward_hook(collector.make_hook("embed_tokens")))
    handles.append(model.norm.register_forward_hook(collector.make_hook("norm")))

    num_layers = len(model.layers)
    for layer_idx, layer in enumerate(model.layers):
        layer_position = (layer_idx + 1) / num_layers
        handles.append(
            layer.self_attn.register_forward_hook(
                collector.make_hook(
                    "self_attn.out",
                    layer_index=layer_idx,
                    layer_position=layer_position,
                )
            )
        )
        handles.append(
            layer.register_forward_hook(
                collector.make_hook(
                    "block.out",
                    layer_index=layer_idx,
                    layer_position=layer_position,
                )
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                collector.make_hook(
                    "mlp.out",
                    layer_index=layer_idx,
                    layer_position=layer_position,
                )
            )
        )
        router = getattr(layer.mlp, "router", None)
        gate = getattr(router, "gate", None)
        if gate is not None:
            handles.append(
                gate.register_forward_hook(
                    collector.make_hook(
                        "router_logits",
                        layer_index=layer_idx,
                        layer_position=layer_position,
                    )
                )
            )
    return handles


def _write_summary_csv(summary: Sequence[CoordCheckSummaryRecord], path: Path) -> None:
    header = (
        "axis,scale,step,family,probe,stat,value,num_runs,num_layers,hidden_size,"
        "intermediate_size,num_attention_heads,num_key_value_heads,num_experts,"
        "num_experts_per_tok,layer_index,layer_position\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for record in summary:
            handle.write(
                ",".join(
                    [
                        str(record.axis),
                        str(record.scale),
                        str(record.step),
                        str(record.family),
                        str(record.probe),
                        str(record.stat),
                        str(record.value),
                        str(record.num_runs),
                        str(record.num_layers),
                        str(record.hidden_size),
                        str(record.intermediate_size),
                        str(record.num_attention_heads),
                        str(record.num_key_value_heads),
                        str(record.num_experts),
                        str(record.num_experts_per_tok),
                        "" if record.layer_index is None else str(record.layer_index),
                        "" if record.layer_position is None else str(record.layer_position),
                    ]
                )
                + "\n"
            )


def _write_summary_charts(
    summary: Sequence[CoordCheckSummaryRecord],
    output_dir: Path,
) -> dict[str, Path]:
    import altair as alt
    import pandas as pd

    dataframe = pd.DataFrame(asdict(record) for record in summary)
    if dataframe.empty:
        return {}

    dataframe["probe_label"] = dataframe.apply(
        lambda row: (
            row["probe"]
            if pd.isna(row["layer_position"])
            else f"{row['probe']}@{float(row['layer_position']):.2f}"
        ),
        axis=1,
    )

    chart_paths: dict[str, Path] = {}
    for axis in ("width", "depth"):
        axis_frame = dataframe[dataframe["axis"] == axis].copy()
        if axis_frame.empty:
            continue
        chart = (
            alt.Chart(axis_frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("scale:O", title=f"{axis.title()} multiplier"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("probe_label:N", title="Probe"),
                tooltip=[
                    "axis",
                    "scale",
                    "step",
                    "family",
                    "probe",
                    "stat",
                    alt.Tooltip("value:Q", format=".6g"),
                    "num_layers",
                    "hidden_size",
                ],
            )
            .properties(width=240, height=140)
            .facet(row="family:N", column="stat:N")
            .resolve_scale(y="independent")
        )
        chart_path = output_dir / f"coord_check_{axis}.html"
        chart.save(chart_path)
        chart_paths[axis] = chart_path
    return chart_paths
