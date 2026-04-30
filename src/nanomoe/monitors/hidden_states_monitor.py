from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from nanomoe.data.packed_dataset import create_document_mask, cu_seqlens_to_packing_metadata
from nanomoe.data.types import PackedBatch
from nanomoe.model.model import MoETransformer
from nanomoe.monitors.attention_monitor import Batch, DatasetLike, _iter_batches, _model_device


@dataclass(frozen=True)
class TokenHiddenStateLayerStats:
    """Average token-to-token cosine similarity inside each sequence."""

    layer_idx: int
    num_sequences: int
    mean_off_diagonal_cosine: float


@dataclass(frozen=True)
class NeighbourLayerHiddenStateStats:
    """Average same-token changes between neighbouring layer outputs."""

    layer_idx: int
    next_layer_idx: int
    num_tokens: int
    mean_token_cosine: float
    mean_rms_distance: float
    mean_relative_rms_change: float


@dataclass(frozen=True)
class RouterUsageLayerStats:
    """Entropy of routed expert assignment frequencies for one layer."""

    layer_idx: int
    num_tokens: int
    num_assignments: int
    entropy: float


@dataclass(frozen=True)
class HiddenStateMonitorResult:
    """Hidden-state cosine summaries collected from transformer layer outputs."""

    intra_sequence: list[TokenHiddenStateLayerStats]
    neighbouring_layers: list[NeighbourLayerHiddenStateStats]
    first_layer_reference: list[NeighbourLayerHiddenStateStats]
    router_usage_entropy: list[RouterUsageLayerStats]
    router_logits: list[Tensor]

    def as_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            "intra_sequence": [item.__dict__.copy() for item in self.intra_sequence],
            "neighbouring_layers": [item.__dict__.copy() for item in self.neighbouring_layers],
            "first_layer_reference": [item.__dict__.copy() for item in self.first_layer_reference],
            "router_usage_entropy": [item.__dict__.copy() for item in self.router_usage_entropy],
        }


def hidden_state_cosine_similarities(
    model: MoETransformer,
    dataset: DatasetLike | Iterable[PackedBatch],
    *,
    batch_size: int | None = None,
    max_batches: int | None = None,
    first_layer_as_reference: bool = False,
) -> HiddenStateMonitorResult:
    """Measure hidden-state cosine similarities on a small dataset.

    This records:
    - average off-diagonal token cosine similarity inside each sequence, per layer;
    - average same-token cosine similarity and RMS changes between neighbouring transformer layers;
    - the same metrics for each layer output against the first transformer layer output.
    """
    del first_layer_as_reference

    device = _model_device(model)
    was_training = model.training
    num_layers = len(model.layers)
    intra_accumulators = [_IntraSequenceAccumulator(layer_idx=i) for i in range(num_layers)]
    neighbour_accumulators = [
        _NeighbourLayerAccumulator(layer_idx=i, next_layer_idx=i + 1) for i in range(num_layers - 1)
    ]
    first_layer_reference_accumulators = [
        _NeighbourLayerAccumulator(layer_idx=0, next_layer_idx=i) for i in range(1, num_layers)
    ]
    router_accumulators = [_RouterUsageAccumulator(layer_idx=i) for i in range(num_layers)]
    router_logits_by_layer: list[list[Tensor]] = [[] for _ in range(num_layers)]

    captured: list[Tensor] = []
    hooks = [layer.register_forward_hook(_make_layer_capture_hook(captured)) for layer in model.layers]
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(_iter_batches(dataset, batch_size=batch_size)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                kwargs = _batch_to_model_kwargs(batch, model=model, device=device)
                captured.clear()
                outputs = model(**kwargs, return_router_logits=True)

                if len(captured) != num_layers:
                    raise RuntimeError(f"Expected {num_layers} captured layer outputs, got {len(captured)}")

                token_mask = _token_mask_from_batch(batch, kwargs, captured[0])
                for layer_idx, hidden_states in enumerate(captured):
                    intra_accumulators[layer_idx].add(hidden_states, token_mask, batch)

                for accumulator in neighbour_accumulators:
                    accumulator.add(
                        captured[accumulator.layer_idx],
                        captured[accumulator.next_layer_idx],
                        token_mask,
                    )

                for accumulator in first_layer_reference_accumulators:
                    accumulator.add(
                        captured[accumulator.layer_idx],
                        captured[accumulator.next_layer_idx],
                        token_mask,
                    )

                _accumulate_router_outputs(
                    outputs.router_logits,
                    outputs.router_expert_indices,
                    token_mask,
                    router_logits_by_layer,
                    router_accumulators,
                )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return HiddenStateMonitorResult(
        intra_sequence=[acc.to_stats() for acc in intra_accumulators],
        neighbouring_layers=[acc.to_stats() for acc in neighbour_accumulators],
        first_layer_reference=[acc.to_stats() for acc in first_layer_reference_accumulators],
        router_usage_entropy=[acc.to_stats() for acc in router_accumulators],
        router_logits=[
            torch.cat(layer_logits, dim=0) if layer_logits else torch.empty(0) for layer_logits in router_logits_by_layer
        ],
    )


class _IntraSequenceAccumulator:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.cosine_sum = 0.0
        self.num_sequences = 0

    def add(self, hidden_states: Tensor, token_mask: Tensor, batch: Batch | PackedBatch) -> None:
        normalized = F.normalize(hidden_states.float(), p=2, dim=-1)
        for seq_states, seq_mask in _iter_sequence_states_and_masks(normalized, token_mask, batch):
            valid_states = seq_states[seq_mask]
            seq_len = int(valid_states.shape[0])
            if seq_len < 2:
                continue
            cosine = valid_states @ valid_states.T
            off_diag = ~torch.eye(seq_len, dtype=torch.bool, device=cosine.device)
            self.cosine_sum += float(cosine[off_diag].mean().item())
            self.num_sequences += 1

    def to_stats(self) -> TokenHiddenStateLayerStats:
        return TokenHiddenStateLayerStats(
            layer_idx=self.layer_idx,
            num_sequences=self.num_sequences,
            mean_off_diagonal_cosine=self.cosine_sum / self.num_sequences if self.num_sequences else float("nan"),
        )


class _NeighbourLayerAccumulator:
    def __init__(self, layer_idx: int, next_layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.next_layer_idx = next_layer_idx
        self.cosine_sum = 0.0
        self.rms_distance_sum = 0.0
        self.relative_rms_change_sum = 0.0
        self.num_tokens = 0

    def add(self, hidden_states: Tensor, next_hidden_states: Tensor, token_mask: Tensor) -> None:
        hidden_states = hidden_states.float()
        next_hidden_states = next_hidden_states.float()
        cosine = F.cosine_similarity(hidden_states, next_hidden_states, dim=-1)
        valid_cosine = cosine[token_mask]
        if valid_cosine.numel() == 0:
            return

        rms = _rms_norm(hidden_states)
        next_rms = _rms_norm(next_hidden_states)
        rms_distance = _rms_norm(next_hidden_states - hidden_states)
        relative_rms_change = (next_rms - rms) / rms.clamp_min(torch.finfo(rms.dtype).eps)

        self.cosine_sum += float(valid_cosine.sum().item())
        self.rms_distance_sum += float(rms_distance[token_mask].sum().item())
        self.relative_rms_change_sum += float(relative_rms_change[token_mask].sum().item())
        self.num_tokens += int(valid_cosine.numel())

    def to_stats(self) -> NeighbourLayerHiddenStateStats:
        return NeighbourLayerHiddenStateStats(
            layer_idx=self.layer_idx,
            next_layer_idx=self.next_layer_idx,
            num_tokens=self.num_tokens,
            mean_token_cosine=self.cosine_sum / self.num_tokens if self.num_tokens else float("nan"),
            mean_rms_distance=self.rms_distance_sum / self.num_tokens if self.num_tokens else float("nan"),
            mean_relative_rms_change=self.relative_rms_change_sum / self.num_tokens if self.num_tokens else float("nan"),
        )


class _RouterUsageAccumulator:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.num_tokens = 0
        self.num_assignments = 0
        self.expert_counts: Tensor | None = None

    def add(self, expert_indices: Tensor, token_mask: Tensor, num_experts: int) -> None:
        valid_indices = expert_indices[token_mask]
        if valid_indices.numel() == 0:
            return

        counts = torch.bincount(valid_indices.reshape(-1), minlength=num_experts).cpu()
        if self.expert_counts is None:
            self.expert_counts = counts
        else:
            self.expert_counts += counts
        self.num_tokens += int(valid_indices.shape[0])
        self.num_assignments += int(valid_indices.numel())

    def to_stats(self) -> RouterUsageLayerStats:
        if self.expert_counts is None or self.num_assignments == 0:
            entropy = float("nan")
        else:
            frequencies = self.expert_counts.float() / self.num_assignments
            nonzero = frequencies > 0
            entropy = float(-(frequencies[nonzero] * frequencies[nonzero].log2()).sum().item())

        return RouterUsageLayerStats(
            layer_idx=self.layer_idx,
            num_tokens=self.num_tokens,
            num_assignments=self.num_assignments,
            entropy=entropy,
        )


def _make_layer_capture_hook(captured: list[Tensor]):
    @torch.compiler.disable
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden_states = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden_states, Tensor):
            raise TypeError(f"Expected transformer layer hook output to contain a Tensor, got {type(hidden_states)!r}")
        captured.append(hidden_states.detach())

    return hook


def _rms_norm(hidden_states: Tensor) -> Tensor:
    return hidden_states.square().mean(dim=-1).sqrt()


def _accumulate_router_outputs(
    router_logits: list[Tensor] | None,
    router_expert_indices: list[Tensor] | None,
    token_mask: Tensor,
    router_logits_by_layer: list[list[Tensor]],
    router_accumulators: list[_RouterUsageAccumulator],
) -> None:
    if router_logits is None or router_expert_indices is None:
        return
    if len(router_logits) == 0 and len(router_expert_indices) == 0:
        return
    if len(router_logits) != len(router_expert_indices):
        raise RuntimeError(
            f"Expected matching router logits and expert index lists, got "
            f"{len(router_logits)} and {len(router_expert_indices)}"
        )
    if len(router_logits) != len(router_accumulators):
        raise RuntimeError(f"Expected router outputs for {len(router_accumulators)} layers, got {len(router_logits)}")

    for layer_idx, (layer_logits, layer_expert_indices) in enumerate(
        zip(router_logits, router_expert_indices, strict=True)
    ):
        valid_mask = token_mask[:, : layer_logits.shape[1]]
        valid_logits = layer_logits[:, : valid_mask.shape[1]][valid_mask]
        router_logits_by_layer[layer_idx].append(valid_logits.detach().cpu())
        router_accumulators[layer_idx].add(
            layer_expert_indices[:, : valid_mask.shape[1]],
            valid_mask,
            num_experts=layer_logits.shape[-1],
        )


def _batch_to_model_kwargs(
    batch: Batch | PackedBatch,
    *,
    model: MoETransformer,
    device: torch.device,
) -> dict[str, Tensor]:
    if isinstance(batch, PackedBatch):
        batch = batch.to(device)
        kwargs = {
            "input_ids": batch.tokens.unsqueeze(0),
            "position_ids": batch.position_ids.unsqueeze(0),
        }
        if model.config.attention_type == "flex_attention":
            doc_ids, seq_lens = cu_seqlens_to_packing_metadata(batch.cu_seqlens)
            kwargs["packing_doc_ids"] = doc_ids
            kwargs["packing_seq_lens"] = seq_lens
        else:
            kwargs["attention_mask"] = create_document_mask(batch.cu_seqlens, dtype=torch.float32)
        return kwargs

    if isinstance(batch, Tensor):
        return {"input_ids": batch.to(device)}

    kwargs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "position_ids", "packing_doc_ids", "packing_seq_lens"}
        and isinstance(value, Tensor)
    }
    if "input_ids" not in kwargs:
        raise ValueError("Monitor batches must provide an input_ids tensor")
    return kwargs


def _iter_sequence_states_and_masks(
    hidden_states: Tensor,
    token_mask: Tensor,
    batch: Batch | PackedBatch,
) -> Iterable[tuple[Tensor, Tensor]]:
    if isinstance(batch, PackedBatch):
        cu_seqlens = batch.cu_seqlens.to(device=hidden_states.device)
        if hidden_states.shape[0] != 1:
            raise ValueError(f"PackedBatch hidden states must have batch size 1, got {hidden_states.shape[0]}")
        for start_t, end_t in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            start = int(start_t.item())
            end = int(end_t.item())
            yield hidden_states[0, start:end], token_mask[0, start:end]
        return

    yield from zip(hidden_states, token_mask, strict=True)


def _token_mask_from_batch(batch: Batch | PackedBatch, kwargs: Mapping[str, Tensor], hidden_states: Tensor) -> Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None and attention_mask.dim() == 2:
        return attention_mask[:, :seq_len].to(device=hidden_states.device, dtype=torch.bool)

    packing_seq_lens = kwargs.get("packing_seq_lens")
    if packing_seq_lens is not None:
        idx = torch.arange(seq_len, device=hidden_states.device).view(1, seq_len)
        return idx < packing_seq_lens.view(batch_size, 1)

    if isinstance(batch, Mapping):
        labels = batch.get("labels")
        if isinstance(labels, Tensor) and labels.shape[:2] == (batch_size, seq_len):
            return labels.to(device=hidden_states.device) != -100

    return torch.ones((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
