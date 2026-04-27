from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from nanomoe.model.model import MoETransformer
from nanomoe.monitors.attention_monitor import Batch, DatasetLike, _batch_to_model_kwargs, _iter_batches, _model_device


@dataclass(frozen=True)
class TokenHiddenStateLayerStats:
    """Average token-to-token cosine similarity inside each sequence."""

    layer_idx: int
    num_sequences: int
    mean_off_diagonal_cosine: float


@dataclass(frozen=True)
class NeighbourLayerHiddenStateStats:
    """Average same-token cosine similarity between neighbouring layer outputs."""

    layer_idx: int
    next_layer_idx: int
    num_tokens: int
    mean_token_cosine: float


@dataclass(frozen=True)
class HiddenStateMonitorResult:
    """Hidden-state cosine summaries collected from transformer layer outputs."""

    intra_sequence: list[TokenHiddenStateLayerStats]
    neighbouring_layers: list[NeighbourLayerHiddenStateStats]

    def as_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            "intra_sequence": [item.__dict__.copy() for item in self.intra_sequence],
            "neighbouring_layers": [item.__dict__.copy() for item in self.neighbouring_layers],
        }


def hidden_state_cosine_similarities(
    model: MoETransformer,
    dataset: DatasetLike,
    *,
    batch_size: int | None = None,
    max_batches: int | None = None,
) -> HiddenStateMonitorResult:
    """Measure hidden-state cosine similarities on a small dataset.

    This records:
    - average off-diagonal token cosine similarity inside each sequence, per layer;
    - average same-token cosine similarity between neighbouring transformer layers.
    """

    device = _model_device(model)
    was_training = model.training
    num_layers = len(model.layers)
    intra_accumulators = [_IntraSequenceAccumulator(layer_idx=i) for i in range(num_layers)]
    neighbour_accumulators = [
        _NeighbourLayerAccumulator(layer_idx=i, next_layer_idx=i + 1) for i in range(max(0, num_layers - 1))
    ]

    captured: list[Tensor] = []
    hooks = [layer.register_forward_hook(_make_layer_capture_hook(captured)) for layer in model.layers]

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(_iter_batches(dataset, batch_size=batch_size)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                kwargs = _batch_to_model_kwargs(batch, device=device)
                captured.clear()
                _ = model(**kwargs)

                if len(captured) != num_layers:
                    raise RuntimeError(f"Expected {num_layers} captured layer outputs, got {len(captured)}")

                token_mask = _token_mask_from_batch(batch, kwargs, captured[0])
                for layer_idx, hidden_states in enumerate(captured):
                    intra_accumulators[layer_idx].add(hidden_states, token_mask)

                for layer_idx in range(num_layers - 1):
                    neighbour_accumulators[layer_idx].add(captured[layer_idx], captured[layer_idx + 1], token_mask)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return HiddenStateMonitorResult(
        intra_sequence=[acc.to_stats() for acc in intra_accumulators],
        neighbouring_layers=[acc.to_stats() for acc in neighbour_accumulators],
    )


class _IntraSequenceAccumulator:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.cosine_sum = 0.0
        self.num_sequences = 0

    def add(self, hidden_states: Tensor, token_mask: Tensor) -> None:
        normalized = F.normalize(hidden_states.float(), p=2, dim=-1)
        for seq_states, seq_mask in zip(normalized, token_mask, strict=True):
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
        self.num_tokens = 0

    def add(self, hidden_states: Tensor, next_hidden_states: Tensor, token_mask: Tensor) -> None:
        cosine = F.cosine_similarity(hidden_states.float(), next_hidden_states.float(), dim=-1)
        valid_cosine = cosine[token_mask]
        if valid_cosine.numel() == 0:
            return
        self.cosine_sum += float(valid_cosine.sum().item())
        self.num_tokens += int(valid_cosine.numel())

    def to_stats(self) -> NeighbourLayerHiddenStateStats:
        return NeighbourLayerHiddenStateStats(
            layer_idx=self.layer_idx,
            next_layer_idx=self.next_layer_idx,
            num_tokens=self.num_tokens,
            mean_token_cosine=self.cosine_sum / self.num_tokens if self.num_tokens else float("nan"),
        )


def _make_layer_capture_hook(captured: list[Tensor]):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden_states = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden_states, Tensor):
            raise TypeError(f"Expected transformer layer hook output to contain a Tensor, got {type(hidden_states)!r}")
        captured.append(hidden_states.detach())

    return hook


def _token_mask_from_batch(batch: Batch, kwargs: Mapping[str, Tensor], hidden_states: Tensor) -> Tensor:
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
