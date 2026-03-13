from __future__ import annotations

from typing import Literal

import chz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

AttentionBackend = Literal["sdpa", "hullattn"]
PoolingMode = Literal["last", "mean", "cls"]


@chz.chz
class TransformerClassifierConfig:
    vocab_size: int | None = None
    num_classes: int = 2
    max_seq_len: int = 2048
    input_mode: Literal["token", "continuous"] = "token"
    input_dim: int = 1
    pad_token_id: int = 0
    d_model: int = 128
    num_layers: int = 8
    num_heads: int = 8
    ffn_hidden_size: int = 512
    dropout: float = 0.1
    attention_backend: AttentionBackend = "sdpa"
    hull_top_k: int = 8
    pooling: PoolingMode = "last"
    use_cls_token: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self.d_model // self.num_heads


def _hull_attention_reference(q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor, top_k: int) -> Tensor:
    """Brute-force 2D top-k sparse attention used as the first hull-attention reference path.

    This matches the candidate-selection semantics we want from `hullattn`, but computes the
    top-k keys by exhaustive search instead of a convex-hull data structure. The current path is
    intentionally correctness-first and only supports 2D keys/queries.
    """

    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q, k, and v must have matching shapes")
    if q.ndim != 4:
        raise ValueError(f"Expected q, k, v to have shape [batch, heads, seq_len, head_dim], got {tuple(q.shape)}")
    if q.shape[-1] != 2:
        raise ValueError(f"hullattn reference path requires head_dim=2, got {q.shape[-1]}")
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask to have shape [batch, seq_len], got {tuple(attention_mask.shape)}")
    if top_k < 1:
        raise ValueError(f"hull_top_k must be >= 1, got {top_k}")

    _, _, seq_len, _ = q.shape
    top_k = min(top_k, seq_len)

    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    key_mask = attention_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

    topk_scores, topk_indices = scores.topk(k=top_k, dim=-1)
    gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, v.shape[-1])
    value_bank = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
    topk_values = torch.take_along_dim(value_bank, gather_index, dim=3)

    weights = torch.softmax(topk_scores.float(), dim=-1).to(dtype=v.dtype)
    attended = (weights.unsqueeze(-1) * topk_values).sum(dim=-2)

    query_mask = attention_mask[:, None, :, None]
    return attended * query_mask.to(dtype=attended.dtype)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: TransformerClassifierConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.config.attention_backend == "hullattn":
            if self.head_dim != 2:
                raise ValueError(
                    "attention_backend='hullattn' currently requires head_dim=2. "
                    "Set num_heads=d_model//2 for the LRA reference implementation."
                )

        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.config.attention_backend == "hullattn":
            attended = _hull_attention_reference(q, k, v, attention_mask, self.config.hull_top_k)
        else:
            key_mask = attention_mask[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        attended = self.out_proj(attended)
        return self.dropout(attended)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerClassifierConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: TransformerClassifierConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = MultiheadSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerClassifierConfig):
        super().__init__()
        self.config = config

        if config.input_mode == "token":
            if config.vocab_size is None:
                raise ValueError("vocab_size is required for token inputs")
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
            self.input_projection = None
        else:
            self.token_embedding = None
            self.input_projection = nn.Linear(config.input_dim, config.d_model)

        self.cls_embedding = nn.Parameter(torch.zeros(config.d_model)) if config.use_cls_token else None
        self.position_embedding = nn.Embedding(config.max_seq_len + int(config.use_cls_token), config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerEncoderBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.token_embedding is not None:
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            if self.token_embedding.padding_idx is not None:
                with torch.no_grad():
                    self.token_embedding.weight[self.token_embedding.padding_idx].zero_()
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            if self.input_projection.bias is not None:
                nn.init.zeros_(self.input_projection.bias)
        if self.cls_embedding is not None:
            nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in {self.classifier, self.input_projection}:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed_inputs(self, inputs: Tensor) -> Tensor:
        if self.config.input_mode == "token":
            assert self.token_embedding is not None
            return self.token_embedding(inputs.squeeze(-1) if inputs.ndim == 3 else inputs)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(-1)
        assert self.input_projection is not None
        return self.input_projection(inputs)

    def _prepend_cls(self, hidden_states: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.cls_embedding is None:
            return hidden_states, attention_mask

        cls_tokens = self.cls_embedding.view(1, 1, -1).expand(hidden_states.shape[0], -1, -1)
        cls_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([cls_tokens, hidden_states], dim=1), torch.cat([cls_mask, attention_mask], dim=1)

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]

        if self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1)
            return summed / denom

        lengths = attention_mask.sum(dim=1).clamp_min(1)
        last_index = lengths - 1
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_index, :]

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = self._embed_inputs(inputs)
        hidden_states, attention_mask = self._prepend_cls(hidden_states, attention_mask)
        seq_len = hidden_states.shape[1]
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self.position_embedding.num_embeddings}"
            )

        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        pooled = self._pool(hidden_states, attention_mask)
        return self.classifier(pooled)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.parameters())


def build_transformer_classifier(config: TransformerClassifierConfig) -> TransformerClassifier:
    return TransformerClassifier(config)
