"""Attention layers with RoPE and GQA support.

Features:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Flash Attention 2 support via F.scaled_dot_product_attention
- Document masking via cu_seqlens (TODO: This is not implemented yet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanomoe.model.config import MoEConfig


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

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Get cos and sin for RoPE.

        Args:
            x: Input tensor [batch, seq_len, ...] for shape/device info
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            cos, sin: [seq_len, dim] or [batch, seq_len, dim] if position_ids provided
        """
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
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
        cos, sin = self.rope(hidden_states, position_ids)
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

        # Convert mask format if needed
        # F.scaled_dot_product_attention expects mask where True = masked (opposite of typical)
        if attention_mask is not None:
            # Our mask: 0 = attend, -inf = masked
            # SDPA mask: True = masked, False = attend
            attn_mask = attention_mask < -1000  # Convert -inf to True
        else:
            # Create causal mask
            attn_mask = None  # SDPA handles causal automatically with is_causal=True

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None,  # Use built-in causal if no custom mask
        )

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, past_key_value
