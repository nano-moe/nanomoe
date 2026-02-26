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
from nanomoe.model.moe_kernel import MOE_KERNEL_REGISTRY
from nanomoe.model.moe_router import ROUTER_REGISTRY, NaiveTopKRouter

TopKRouter = NaiveTopKRouter


class SwiGLU(nn.Module):
    """SwiGLU activation: gate * silu(x) * up(x)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> Tensor:
        del packing_doc_ids, packing_seq_lens
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """Packed expert weights compatible with moe_kernel API."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        has_bias: bool = False,
        is_transposed: bool = False,
        kernel: str = "grouped_mm",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.has_bias = has_bias
        self.is_transposed = is_transposed
        self.kernel_fn = MOE_KERNEL_REGISTRY[kernel]

        gate_up_out = 2 * intermediate_size
        if is_transposed:
            self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, gate_up_out))
            self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        else:
            self.gate_up_proj = nn.Parameter(torch.empty(num_experts, gate_up_out, hidden_size))
            self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

        if has_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(num_experts, gate_up_out))
            self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_size))
        else:
            self.register_parameter("gate_up_proj_bias", None)
            self.register_parameter("down_proj_bias", None)

        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)
        if self.has_bias:
            nn.init.zeros_(self.gate_up_proj_bias)
            nn.init.zeros_(self.down_proj_bias)

    def _apply_gate(self, gate_up_out: Tensor) -> Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> Tensor:
        del packing_doc_ids, packing_seq_lens
        return self.kernel_fn(self, hidden_states, top_k_index, top_k_weights)


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.router = ROUTER_REGISTRY[config.router_type](config)

        # Experts
        kernel = config.moe_kernel
        if kernel == "auto":
            kernel = "grouped_mm" if hasattr(F, "grouped_mm") else "eager_mm"
        if kernel not in MOE_KERNEL_REGISTRY:
            raise ValueError(f"Unsupported moe_kernel: {kernel}. Available: {list(MOE_KERNEL_REGISTRY)}")
        if kernel != "eager_mm" and not hasattr(F, "grouped_mm"):
            raise ValueError(
                f"Requested moe_kernel={kernel}, but torch.nn.functional.grouped_mm is unavailable. "
                "Use moe_kernel='eager_mm' instead."
            )
        self.moe_kernel = kernel
        self.experts = Expert(
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            has_bias=False,
            is_transposed=False,
            kernel=self.moe_kernel,
            init_std=config.initializer_range,
        )

        # Shared expert (always active, if enabled)
        self.shared_expert = None
        if config.shared_expert:
            shared_size = config.shared_expert_intermediate_size or config.intermediate_size
            self.shared_expert = SwiGLU(config.hidden_size, shared_size)

    def forward(
        self,
        hidden_states: Tensor,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
        router_logits, expert_indices, expert_weights = self.router(
            hidden_states_flat,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )

        # Compute auxiliary loss
        aux_loss = self.router.compute_aux_loss(router_logits)

        # Compute expert outputs
        final_output = self.experts(
            hidden_states_flat,
            expert_indices,
            expert_weights,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(
                hidden_states_flat,
                packing_doc_ids=packing_doc_ids,
                packing_seq_lens=packing_seq_lens,
            )
            # Shared expert gets equal weight to one routed expert
            final_output = final_output + shared_output * self.config.shared_expert_scale
            # Rescale routed experts
            # final_output = final_output
            #  * (self.num_experts_per_tok + 1) / self.num_experts_per_tok

        return final_output.view(batch_size, seq_len, hidden_size), aux_loss


class DenseFFN(nn.Module):
    """Dense (non-MoE) FFN layer using SwiGLU."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        # Use intermediate_size * num_experts_per_tok to match active params
        intermediate = config.intermediate_size * config.num_experts_per_tok
        self.ffn = SwiGLU(config.hidden_size, intermediate)

    def forward(
        self,
        hidden_states: Tensor,
        packing_doc_ids: Tensor | None = None,
        packing_seq_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns output and zero aux_loss (for API compatibility with MoE).
        """
        return (
            self.ffn(
                hidden_states,
                packing_doc_ids=packing_doc_ids,
                packing_seq_lens=packing_seq_lens,
            ),
            torch.tensor(0.0, device=hidden_states.device),
        )
