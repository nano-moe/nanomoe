"""
Moe kernel implementations.

Partially adapted from:
https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py
"""

import torch
import torch.nn.functional as F


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Grouped linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, output_dim, input_dim) if transposed is `False`,
            else of shape (num_experts, input_dim, output_dim).
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (num_experts, output_dim). Default is `None`.
        offs (`torch.Tensor`, *optional*):
            Offsets tensor indicating the boundaries of each group in the input tensor.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """
    if is_transposed:
        # (S, input_dim) @ grouped (num_experts, input_dim, output_dim) -> (S, output_dim)
        out = F.grouped_mm(input, weight, offs=offs)
    else:
        # (S, input_dim) @ grouped (num_experts, output_dim, input_dim).T -> (S, output_dim)
        out = F.grouped_mm(input, weight.transpose(-2, -1), offs=offs)

    if bias is not None:
        # We should be able to pass bias to the grouped_mm call, but it's not yet supported.
        out = out + bias

    return out


def eager_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Select expert weights and biases for selected samples
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = self.gate_up_proj_bias if self.has_bias else None
    selected_down_bias = self.down_proj_bias if self.has_bias else None
    gate_up_fn = torch.matmul if self.is_transposed else lambda x, y: torch.matmul(x, y.transpose(0, 1))
    down_fn = torch.matmul if self.is_transposed else lambda x, y: torch.matmul(x, y.transpose(0, 1))
    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    start_idx = 0
    outputs = []
    for i, num_token in enumerate(num_tokens_per_expert):
        end_idx = start_idx + int(num_token)
        if num_token == 0:
            continue
        gate_up, gate_down = selected_gate_up[i], selected_down[i]
        gate_up_bias = selected_gate_up_bias[i] if selected_gate_up_bias is not None else 0
        gate_down_bias = selected_down_bias[i] if selected_down_bias is not None else 0
        tokens_for_this_expert = selected_hidden_states_g[start_idx:end_idx]
        expert_out = self._apply_gate(gate_up_fn(tokens_for_this_expert, gate_up) + gate_up_bias)
        expert_out = down_fn(expert_out, gate_down) + gate_down_bias
        outputs.append(expert_out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if len(outputs) else selected_hidden_states_g.new_empty(0)
    outs = outs * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
    outs = outs[inv_perm]
    final_out = outs.view(*top_k_index.shape, -1).type(top_k_weights.dtype).sum(dim=1).type(outs.dtype)
    return final_out


def grouped_mm_experts_forward_fast(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    A faster grouped mm based implementation,
    risking non-determinism due to use of atomicAdd in the accumulation step.
    """
    if not hasattr(F, "grouped_mm"):
        raise ImportError(
            "F.grouped_mm is not available. Please make sure you are using a PyTorch version that includes it (2.9+)."
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).repeat_interleave(num_top_k)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    token_idx_g = token_idx[perm]

    selected_hidden_states_g = hidden_states.index_select(0, token_idx_g)

    # Select expert weights and biases for selected samples
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (grouped) ---
    gate_up_out = _grouped_linear(
        selected_hidden_states_g, selected_gate_up, selected_gate_up_bias, offsets, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (grouped) ---
    out_per_sample_g = _grouped_linear(
        gated_out, selected_down, selected_down_bias, offsets, is_transposed=self.is_transposed
    )  # (S, hidden_dim)
    # Apply routing weights, avoid in-place ops to prevent type promotion issues
    # as sample_weights_g could be fp32 while out_per_sample_g is bf16/fp16
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)
    # Restore original order
    final = torch.zeros(num_tokens, hidden_dim, device=out_per_sample_g.device, dtype=out_per_sample_g.dtype)
    final.index_add_(0, token_idx_g, out_per_sample_g)  # Source of non-deterministic
    return final.to(hidden_states.dtype)


def grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(F, "grouped_mm"):
        raise ImportError(
            "F.grouped_mm is not available. Please make sure you are using a PyTorch version that includes it (2.9+)."
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Select expert weights and biases for selected samples
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (grouped) ---
    gate_up_out = _grouped_linear(
        selected_hidden_states_g, selected_gate_up, selected_gate_up_bias, offsets, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (grouped) ---
    out_per_sample_g = _grouped_linear(
        gated_out, selected_down, selected_down_bias, offsets, is_transposed=self.is_transposed
    )  # (S, hidden_dim)
    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)
    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


MOE_KERNEL_REGISTRY = {
    "eager_mm": eager_mm_experts_forward,
    "grouped_mm_fast": grouped_mm_experts_forward_fast,
    "grouped_mm": grouped_mm_experts_forward,
}
