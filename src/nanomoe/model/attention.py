"""Attention layers with RoPE and GQA support.

Features:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Flash Attention 2 support via F.scaled_dot_product_attention
- Document masking via cu_seqlens
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

if TYPE_CHECKING:
    from nanomoe.model.config import MoEConfig

# Adopted from https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/masks/document_mask.py


def get_causal_doc_mask(doc_id: Tensor, lengths: Tensor):
    """
    doc_id: [batch, seq_len] document ID for each token
    lengths: [batch] actual sequence length for each batch item (for padding)
    """

    def _doc_causal_mask(b, h, q_idx, kv_idx):
        del h
        q_ok = q_idx < lengths[b]
        kv_ok = kv_idx < lengths[b]
        same_doc = doc_id[b, q_idx] == doc_id[b, kv_idx]
        causal = q_idx >= kv_idx
        return q_ok & kv_ok & same_doc & causal

    return _doc_causal_mask


# Backward-compatible alias for previous typo.
get_casual_doc_mask = get_causal_doc_mask


class RoPE(nn.Module):
    """Rotary Position Embeddings."""

    inv_freq: Tensor  # Declared for type checker

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get cos and sin for RoPE.

        Args:
            x: Input tensor [batch, seq_len, ...] for shape/device info
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            cos, sin: [seq_len, dim] or [batch, seq_len, dim] if position_ids provided
        """
        del packing_doc_ids, packing_seq_lens
        if position_ids is not None:
            seq_len = int(position_ids.max().item()) + 1
        else:
            seq_len = int(x.shape[1]) if x.dim() > 2 else int(x.shape[0])
        self._update_cache(seq_len, x.device, x.dtype)

        # After _update_cache, these are guaranteed to be set
        assert self._cos_cached is not None and self._sin_cached is not None

        if position_ids is None:
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        # Gather cos/sin by position_ids
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim] or [batch, seq_len, head_dim]
        sin: [seq_len, head_dim] or [batch, seq_len, head_dim]

    Returns:
        q_embed, k_embed with RoPE applied
    """
    # Expand cos/sin for broadcasting
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _fsdp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    position_ids: Tensor | None,
    packing_doc_ids: Tensor | None,
    packing_seq_lens: Tensor | None,
) -> Tensor:
    """Wrapper for F.scaled_dot_product_attention to handle mask format and causal logic."""
    del position_ids, packing_doc_ids, packing_seq_lens
    # Convert additive 4D masks into the boolean mask format expected by SDPA.
    if attn_mask is not None:
        # Packed 4D mask uses additive values: 0 for attend and -inf for masked.
        # For boolean SDPA masks, True means attend and False means masked.
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask == 0

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


def _flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    position_ids: Tensor | None,
    packing_doc_ids: Tensor | None,
    packing_seq_lens: Tensor | None,
) -> Tensor:
    """Wrapper for flex_attention to handle mask format and causal logic."""
    if packing_doc_ids is None or packing_seq_lens is None:
        raise ValueError("flex_attention requires packing_doc_ids and packing_seq_lens")

    mask_mod = get_causal_doc_mask(packing_doc_ids, packing_seq_lens)
    block_mask = create_block_mask(
        mask_mod,
        B=int(q.shape[0]),
        H=None,
        Q_LEN=int(q.shape[2]),
        KV_LEN=int(k.shape[2]),
        device=q.device,
    )
    return cast(
        Tensor,
        flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
        ),
    )


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE support."""

    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # head_dim is always set by MoEConfig.__post_init__ if None
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rope = RoPE(self.head_dim, config.max_position_embeddings, config.rope_theta)

        # Dropout
        self.attention_dropout = config.attention_dropout

        # Attention function
        attn_type = getattr(config, "attention_type", "fsdp_attention")
        if attn_type == "fsdp_attention":
            self.attention_fn = _fsdp_attention
        elif attn_type == "flex_attention":
            self.attention_fn = _flex_attention
        else:
            raise ValueError(f"Unsupported attention_type: {attn_type}")

    @torch.compile
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        packing_doc_ids: Tensor | None = None,  # For packed attention, shape [batch, seq_len]
        packing_seq_lens: Tensor | None = None,  # For packed attention, shape [batch]
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, kv_seq_len] or None
            position_ids: [batch, seq_len] or None
            past_key_value: Cached (K, V) for incremental decoding
            use_cache: Whether to return updated KV cache

        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Updated KV cache or None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(
            hidden_states,
            position_ids,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )
        q, k = apply_rope(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # Expand K, V for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        # Use F.scaled_dot_product_attention for efficiency (Flash Attention when available)
        dropout_p = self.attention_dropout if self.training else 0.0

        output = self.attention_fn(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None and not (past_key_value is not None and seq_len == 1),
            position_ids=position_ids,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, past_key_value
