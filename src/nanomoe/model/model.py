"""MoE Transformer model implementation.

A scalable Mixture of Experts transformer based on Qwen3 architecture:
- RMSNorm normalization
- GQA attention with RoPE
- SwiGLU MoE FFN with top-k routing
- Optional shared expert
"""

import torch
import torch.nn as nn
from torch import Tensor

from nanomoe.model.attention import Attention
from nanomoe.model.config import MoEConfig
from nanomoe.model.moe import DenseFFN, MoELayer
from nanomoe.model.output import ModelOutput


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: Tensor,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> Tensor:
        del packing_doc_ids, packing_seq_lens
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MoE FFN."""

    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.residual_scale = config.num_layers ** (-config.depth_alpha)

        # Attention
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # FFN (MoE or Dense)
        if config.num_experts > 1:
            self.mlp = MoELayer(config)
        else:
            self.mlp = DenseFFN(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass.

        Returns:
            hidden_states: Output hidden states
            aux_loss: MoE auxiliary loss
            past_key_value: KV cache if use_cache=True
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.residual_scale * hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.mlp(
            hidden_states,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )
        hidden_states = residual + self.residual_scale * hidden_states

        return hidden_states, aux_loss, past_key_value


class MoETransformer(nn.Module):
    """MoE Transformer Language Model.

    A decoder-only transformer with:
    - Token embeddings
    - N transformer blocks with MoE FFN
    - Final RMSNorm
    - LM head (optionally tied to embeddings)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = None  # Use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, 1, seq_len, kv_seq_len] attention mask
            position_ids: [batch, seq_len] position IDs (for packed sequences)
            packing_doc_ids: [batch, seq_len] document IDs for packed sequences
            packing_seq_lens: [batch] packed sequence lengths
            past_key_values: KV cache for incremental decoding
            use_cache: Whether to return KV cache
            return_dict: Always True (for compatibility)

        Returns:
            ModelOutput with logits, aux_loss, and optional KV cache.
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Default position IDs
        if position_ids is None:
            past_len = 0
            if past_key_values:
                first_kv = next((kv for kv in past_key_values if kv is not None), None)
                if first_kv is not None:
                    past_len = int(first_kv[0].shape[2])
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            )

        # For flex attention, default to one document per batch element if packing metadata
        # is not provided by the caller.
        if self.config.attention_type == "flex_attention" and packing_doc_ids is None and packing_seq_lens is None:
            packing_doc_ids = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
            packing_seq_lens = torch.full((batch_size,), seq_len, device=input_ids.device, dtype=torch.long)

        # Initialize KV cache list
        kv_cache: list[tuple[Tensor, Tensor] | None] = (
            list(past_key_values) if past_key_values else [None] * len(self.layers)
        )
        if len(kv_cache) != len(self.layers):
            raise ValueError(
                f"past_key_values must have one entry per layer, got {len(kv_cache)} for {len(self.layers)} layers"
            )
        new_key_values: list[tuple[Tensor, Tensor] | None] = []

        # Transformer blocks
        total_aux_loss = 0.0
        for i, layer in enumerate(self.layers):
            hidden_states, aux_loss, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packing_doc_ids=packing_doc_ids,
                packing_seq_lens=packing_seq_lens,
                past_key_value=kv_cache[i],
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss
            new_key_values.append(past_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states @ self.embed_tokens.weight.T  # tie word embeddings

        return ModelOutput(
            logits=logits,
            aux_loss=total_aux_loss,
            past_key_values=new_key_values if use_cache else None,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Simple autoregressive generation.

        Args:
            input_ids: [batch, prompt_len] prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (overrides top_p if set)
            eos_token_id: Stop generation at this token

        Returns:
            generated: [batch, prompt_len + generated_len] tokens
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # KV cache
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass (only last token if using cache)
            if past_key_values is None:
                curr_input = generated
                position_ids = torch.arange(generated.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                curr_input = generated[:, -1:]
                position_ids = torch.full((batch_size, 1), generated.shape[1] - 1, device=device)

            outputs = self.forward(
                curr_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
            )

            logits = outputs.logits[:, -1, :]  # [batch, vocab]
            past_key_values = outputs.past_key_values

            # Sample
            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    # Top-k sampling
                    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
                    probs = torch.softmax(top_logits, dim=-1)
                    sample_idx = torch.multinomial(probs, 1)
                    next_token = top_indices.gather(-1, sample_idx)
                else:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Mask tokens beyond top_p
                    mask = cumsum_probs - sorted_probs > top_p
                    sorted_logits[mask] = float("-inf")

                    probs = torch.softmax(sorted_logits, dim=-1)
                    sample_idx = torch.multinomial(probs, 1)
                    next_token = sorted_indices.gather(-1, sample_idx)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return generated

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(config: MoEConfig | str = "small") -> MoETransformer:
    """Create a MoE transformer model.

    Args:
        config: MoEConfig instance or preset name ("tiny", "small", "medium", "large")

    Returns:
        MoETransformer model
    """
    if isinstance(config, str):
        presets = {
            "tiny": MoEConfig.tiny,
            "small": MoEConfig.small,
            "medium": MoEConfig.medium,
            "large": MoEConfig.large,
        }
        if config not in presets:
            raise ValueError(f"Unknown preset: {config}. Available: {list(presets.keys())}")
        config = presets[config]()

    return MoETransformer(config)
