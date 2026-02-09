"""Mixture of Experts layer implementation.

Features:
- Top-k routing with auxiliary load balancing loss
- SwiGLU activation (gate * up * silu)
- Optional shared expert
- Efficient batched expert computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanomoe.model.config import MoEConfig
from nanomoe.model.moe_router import NaiveTopKRouter


class SwiGLU(nn.Module):
    """SwiGLU activation: gate * silu(x) * up(x)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """Single expert FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKRouter(NaiveTopKRouter):
    """Backwards-compatible alias for the default naive top-k router."""


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.router = TopKRouter(config)

        # Experts
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )

        # Shared expert (always active, if enabled)
        self.shared_expert = None
        if config.shared_expert:
            shared_size = config.shared_expert_intermediate_size or config.intermediate_size
            self.shared_expert = Expert(config.hidden_size, shared_size)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through MoE layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: Scalar auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Route tokens to experts
        router_logits, expert_indices, expert_weights = self.router(hidden_states_flat)

        # Compute auxiliary loss
        aux_loss = self.router.compute_aux_loss(router_logits)

        # Compute expert outputs
        # For simplicity, we use a loop over experts
        # In production, use grouped GEMM or token-expert batching
        final_output = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = hidden_states_flat[token_indices]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert
            # expert_indices: [num_tokens, num_experts_per_tok]
            # We need to find which slot (0 to num_experts_per_tok-1) has this expert
            slot_mask = expert_indices[token_indices] == expert_idx
            weights = (expert_weights[token_indices] * slot_mask.float()).sum(dim=-1, keepdim=True)

            # Accumulate weighted output
            final_output[token_indices] += weights * expert_output

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            # Shared expert gets equal weight to one routed expert
            final_output = final_output + shared_output / (self.num_experts_per_tok + 1)
            # Rescale routed experts
            final_output = final_output * (self.num_experts_per_tok + 1) / self.num_experts_per_tok

        return final_output.view(batch_size, seq_len, hidden_size), aux_loss


class DenseFFN(nn.Module):
    """Dense (non-MoE) FFN layer using SwiGLU."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        # Use intermediate_size * num_experts_per_tok to match active params
        intermediate = config.intermediate_size * config.num_experts_per_tok
        self.ffn = SwiGLU(config.hidden_size, intermediate)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns output and zero aux_loss (for API compatibility with MoE).
        """
        return self.ffn(hidden_states), torch.tensor(0.0, device=hidden_states.device)
