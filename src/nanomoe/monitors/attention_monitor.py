from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from nanomoe.model.model import MoETransformer, TransformerBlock

Batch = Tensor | Mapping[str, Any]
DatasetLike = Tensor | Iterable[Batch]

_FORWARD_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "packing_doc_ids",
    "packing_seq_lens",
}


@dataclass(frozen=True)
class AttentionLogitLayerStats:
    """Summary statistics for one layer's pre-softmax attention logits."""

    layer_idx: int
    num_batches: int
    num_logits: int
    mean_query_l2_norm: float
    rms_logit: float
    mean_abs_logit: float
    max_abs_logit: float


@dataclass(frozen=True)
class AttentionLogitMonitorResult:
    """Attention logit norm summaries for every monitored transformer layer."""

    layers: list[AttentionLogitLayerStats]

    def as_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {"layers": [layer.__dict__.copy() for layer in self.layers]}


def attention_logit_norms(
    model: MoETransformer,
    dataset: DatasetLike,
    *,
    batch_size: int | None = None,
    max_batches: int | None = None,
) -> AttentionLogitMonitorResult:
    """Measure pre-softmax attention logit norms on a small dataset.

    Args:
        model: A ``MoETransformer`` instance.
        dataset: Either a tensor of ``input_ids`` with shape ``[batch, seq]`` or
            an iterable yielding tensors or mappings accepted by ``model.forward``.
        batch_size: Optional batch size used only when ``dataset`` is a tensor.
        max_batches: Optional cap for quick probes.

    Returns:
        Per-layer summaries over valid causal/document-masked attention logits.
    """

    device = _model_device(model)
    was_training = model.training
    accumulators = [_AttentionAccumulator(layer_idx=i) for i in range(len(model.layers))]

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(_iter_batches(dataset, batch_size=batch_size)):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                kwargs = _batch_to_model_kwargs(batch, device=device)
                _accumulate_attention_logits(model, kwargs, accumulators)
    finally:
        model.train(was_training)

    return AttentionLogitMonitorResult(layers=[acc.to_stats() for acc in accumulators])


class _AttentionAccumulator:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.num_batches = 0
        self.num_logits = 0
        self.query_l2_sum = 0.0
        self.query_count = 0
        self.square_sum = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0

    def add(self, logits: Tensor, valid_mask: Tensor) -> None:
        logits = logits.float()
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        valid_logits = logits.masked_select(valid_mask)
        if valid_logits.numel() == 0:
            return

        masked_logits = logits.masked_fill(~valid_mask, 0.0)
        query_l2 = torch.linalg.vector_norm(masked_logits, ord=2, dim=-1)

        self.num_batches += 1
        self.num_logits += int(valid_logits.numel())
        self.query_l2_sum += float(query_l2.sum().item())
        self.query_count += int(query_l2.numel())
        self.square_sum += float(valid_logits.square().sum().item())
        self.abs_sum += float(valid_logits.abs().sum().item())
        self.max_abs = max(self.max_abs, float(valid_logits.abs().max().item()))

    def to_stats(self) -> AttentionLogitLayerStats:
        return AttentionLogitLayerStats(
            layer_idx=self.layer_idx,
            num_batches=self.num_batches,
            num_logits=self.num_logits,
            mean_query_l2_norm=self.query_l2_sum / self.query_count if self.query_count else float("nan"),
            rms_logit=math.sqrt(self.square_sum / self.num_logits) if self.num_logits else float("nan"),
            mean_abs_logit=self.abs_sum / self.num_logits if self.num_logits else float("nan"),
            max_abs_logit=self.max_abs if self.num_logits else float("nan"),
        )


def _accumulate_attention_logits(
    model: MoETransformer,
    kwargs: Mapping[str, Tensor],
    accumulators: list[_AttentionAccumulator],
) -> None:
    input_ids = kwargs["input_ids"]
    hidden_states = model.embed_tokens(input_ids)

    batch_size, seq_len = input_ids.shape
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

    attention_mask = kwargs.get("attention_mask")
    packing_doc_ids = kwargs.get("packing_doc_ids")
    packing_seq_lens = kwargs.get("packing_seq_lens")

    for layer_idx, layer_module in enumerate(model.layers):
        layer = layer_module
        if not isinstance(layer, TransformerBlock):
            raise TypeError(f"Expected TransformerBlock at model.layers[{layer_idx}], got {type(layer)!r}")

        attn_input = layer.input_layernorm(hidden_states)
        q, k = _project_attention_qk(layer, attn_input, position_ids, packing_doc_ids, packing_seq_lens)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(layer.self_attn.head_dim)
        valid_mask = _attention_valid_mask(
            logits=logits,
            attention_mask=attention_mask,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )
        accumulators[layer_idx].add(logits, valid_mask)

        hidden_states, _, _ = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
            use_cache=False,
        )


def _project_attention_qk(
    layer: TransformerBlock,
    hidden_states: Tensor,
    position_ids: Tensor | None,
    packing_doc_ids: Tensor | None,
    packing_seq_lens: Tensor | None,
) -> tuple[Tensor, Tensor]:
    attn = layer.self_attn
    batch_size, seq_len, _ = hidden_states.shape
    q = attn.q_proj(hidden_states)
    k = attn.k_proj(hidden_states)

    q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    q = attn.q_norm(q)
    k = attn.k_norm(k)

    cos, sin = attn.rope(
        hidden_states,
        position_ids,
        packing_doc_ids=packing_doc_ids,
        packing_seq_lens=packing_seq_lens,
    )
    q, k = _apply_rope(q, k, cos, sin)

    if attn.num_kv_groups > 1:
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)
    return q, k


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    from nanomoe.model.attention import apply_rope

    return apply_rope(q, k, cos, sin)


def _attention_valid_mask(
    *,
    logits: Tensor,
    attention_mask: Tensor | None,
    packing_doc_ids: Tensor | None,
    packing_seq_lens: Tensor | None,
) -> Tensor:
    batch_size, num_heads, seq_len, kv_seq_len = logits.shape
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            mask = attention_mask
        else:
            mask = attention_mask == 0
        return mask.to(device=logits.device).expand(batch_size, num_heads, seq_len, kv_seq_len)

    q_idx = torch.arange(seq_len, device=logits.device).view(1, 1, seq_len, 1)
    kv_idx = torch.arange(kv_seq_len, device=logits.device).view(1, 1, 1, kv_seq_len)
    mask = q_idx >= kv_idx

    if packing_doc_ids is not None:
        q_doc_ids = packing_doc_ids[:, :seq_len].view(batch_size, 1, seq_len, 1)
        kv_doc_ids = packing_doc_ids[:, :kv_seq_len].view(batch_size, 1, 1, kv_seq_len)
        mask = mask & (q_doc_ids == kv_doc_ids)

    if packing_seq_lens is not None:
        q_ok = q_idx < packing_seq_lens.view(batch_size, 1, 1, 1)
        kv_ok = kv_idx < packing_seq_lens.view(batch_size, 1, 1, 1)
        mask = mask & q_ok & kv_ok

    return mask.expand(batch_size, num_heads, seq_len, kv_seq_len)


def _iter_batches(dataset: DatasetLike, *, batch_size: int | None) -> Iterator[Batch]:
    if isinstance(dataset, Tensor):
        if dataset.dim() != 2:
            raise ValueError(f"Tensor dataset must have shape [batch, seq], got {tuple(dataset.shape)}")
        if batch_size is None:
            yield dataset
            return
        for start in range(0, dataset.shape[0], batch_size):
            yield dataset[start : start + batch_size]
        return

    yield from dataset


def _batch_to_model_kwargs(batch: Batch, *, device: torch.device) -> dict[str, Tensor]:
    if isinstance(batch, Tensor):
        return {"input_ids": batch.to(device)}
    kwargs = {
        key: value.to(device) for key, value in batch.items() if key in _FORWARD_KEYS and isinstance(value, Tensor)
    }
    if "input_ids" not in kwargs:
        raise ValueError("Monitor batches must provide an input_ids tensor")
    return kwargs


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
