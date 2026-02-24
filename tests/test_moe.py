"""
Tests for MoE kernel implementations.
For running full test set:
NANOMOE_RUN_PERF=1 pytest -s tests/test_moe.py
"""

from __future__ import annotations

import copy
import os
import time

import pytest
import torch
import torch.nn.functional as F

from nanomoe.model.moe_kernel import (
    eager_mm_experts_forward,
    grouped_mm_experts_forward,
    grouped_mm_experts_forward_fast,
)


class ToyMoEKernel(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        has_bias: bool,
        is_transposed: bool,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.has_bias = has_bias
        self.is_transposed = is_transposed

        gate_up_out = 2 * intermediate_dim

        if is_transposed:
            self.gate_up_proj = torch.nn.Parameter(torch.randn(num_experts, hidden_dim, gate_up_out))
            self.down_proj = torch.nn.Parameter(torch.randn(num_experts, intermediate_dim, hidden_dim))
        else:
            self.gate_up_proj = torch.nn.Parameter(torch.randn(num_experts, gate_up_out, hidden_dim))
            self.down_proj = torch.nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))

        if has_bias:
            self.gate_up_proj_bias = torch.nn.Parameter(torch.randn(num_experts, gate_up_out))
            self.down_proj_bias = torch.nn.Parameter(torch.randn(num_experts, hidden_dim))
        else:
            self.register_parameter("gate_up_proj_bias", None)
            self.register_parameter("down_proj_bias", None)

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return F.silu(gate) * up


def _make_topk(
    num_tokens: int,
    num_experts: int,
    num_top_k: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.rand(num_tokens, num_experts, device=device)
    topk_weights, topk_indices = torch.topk(scores, num_top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_indices, topk_weights


def _call_eager_or_skip(
    module: ToyMoEKernel,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    try:
        out = eager_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    except Exception as exc:  # pragma: no cover - best-effort detection
        pytest.skip(f"eager_mm_experts_forward unavailable: {exc}")
    if out is None:
        pytest.skip("eager_mm_experts_forward not implemented yet")
    return out


@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_grouped_mm_matches_eager_forward_backward(is_transposed: bool, has_bias: bool) -> None:
    if not hasattr(F, "grouped_mm"):
        pytest.skip("F.grouped_mm is not available")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_experts = 16
    num_top_k = 4
    num_tokens = 16
    hidden_dim = 32
    intermediate_dim = 64

    module = ToyMoEKernel(
        num_experts,
        hidden_dim,
        intermediate_dim,
        has_bias=has_bias,
        is_transposed=is_transposed,
    ).to(device)
    module_ref = copy.deepcopy(module)

    hidden = torch.randn(num_tokens, hidden_dim, device=device, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)

    top_k_index, top_k_weights = _make_topk(num_tokens, num_experts, num_top_k, device=device)
    out_grouped = grouped_mm_experts_forward(module, hidden, top_k_index, top_k_weights)
    out_ref = eager_mm_experts_forward(module_ref, hidden_ref, top_k_index, top_k_weights)
    torch.testing.assert_close(out_grouped, out_ref, rtol=1e-4, atol=1e-4)

    loss_grouped = out_grouped.pow(2).mean()
    loss_ref = out_ref.pow(2).mean()

    loss_grouped.backward()
    loss_ref.backward()

    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=1e-4, atol=1e-4)

    ref_params = dict(module_ref.named_parameters())
    for name, param in module.named_parameters():
        torch.testing.assert_close(
            param.grad,
            ref_params[name].grad,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_grouped_mm_fast_matches_eager_and_grouped_on_cuda(is_transposed: bool, has_bias: bool) -> None:
    if not hasattr(F, "grouped_mm"):
        pytest.skip("F.grouped_mm is not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(1)
    device = torch.device("cuda")

    num_experts = 8
    num_top_k = 4
    num_tokens = 12
    hidden_dim = 24
    intermediate_dim = 40

    module = ToyMoEKernel(
        num_experts,
        hidden_dim,
        intermediate_dim,
        has_bias=has_bias,
        is_transposed=is_transposed,
    ).to(device)
    module_ref = copy.deepcopy(module)
    module_grouped = copy.deepcopy(module)

    hidden = torch.randn(num_tokens, hidden_dim, device=device, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    hidden_grouped = hidden.detach().clone().requires_grad_(True)

    top_k_index, top_k_weights = _make_topk(num_tokens, num_experts, num_top_k, device=device)
    out_grouped_fast = grouped_mm_experts_forward_fast(module, hidden, top_k_index, top_k_weights)
    out_grouped = grouped_mm_experts_forward(module_grouped, hidden_grouped, top_k_index, top_k_weights)
    out_ref = eager_mm_experts_forward(module_ref, hidden_ref, top_k_index, top_k_weights)
    torch.testing.assert_close(out_grouped_fast, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_grouped_fast, out_grouped, rtol=1e-4, atol=1e-4)

    loss_grouped = out_grouped_fast.pow(2).mean()
    loss_ref = out_ref.pow(2).mean()
    loss_grouped_ref = out_grouped.pow(2).mean()

    loss_grouped.backward()
    loss_ref.backward()
    loss_grouped_ref.backward()

    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden.grad, hidden_grouped.grad, rtol=1e-4, atol=1e-4)

    ref_params = dict(module_ref.named_parameters())
    for name, param in module.named_parameters():
        torch.testing.assert_close(
            param.grad,
            ref_params[name].grad,
            rtol=1e-4,
            atol=1e-4,
        )
    grouped_params = dict(module_grouped.named_parameters())
    for name, param in module.named_parameters():
        torch.testing.assert_close(
            param.grad,
            grouped_params[name].grad,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(os.getenv("NANOMOE_RUN_PERF") != "1", reason="perf tests opt-in")
def test_grouped_mm_efficiency_smoke() -> None:
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm is not available")
    if not hasattr(F, "grouped_mm"):
        pytest.skip("F.grouped_mm is not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(2)
    device = torch.device("cuda")

    num_experts = 128
    num_top_k = 8
    num_tokens = 1024
    hidden_dim = 256
    intermediate_dim = 512

    module = ToyMoEKernel(
        num_experts,
        hidden_dim,
        intermediate_dim,
        has_bias=True,
        is_transposed=False,
    ).to(device)

    hidden = torch.randn(num_tokens, hidden_dim, device=device)
    top_k_index, top_k_weights = _make_topk(num_tokens, num_experts, num_top_k, device=device)

    def run_grouped() -> None:
        grouped_mm_experts_forward(module, hidden, top_k_index, top_k_weights)

    def run_grouped_fast() -> None:
        grouped_mm_experts_forward_fast(module, hidden, top_k_index, top_k_weights)

    def run_naive() -> None:
        eager_mm_experts_forward(module, hidden, top_k_index, top_k_weights)

    iters = 5
    with torch.no_grad():
        run_grouped()
        run_grouped_fast()
        run_naive()

        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            run_grouped()
        torch.cuda.synchronize()
        grouped_time = (time.perf_counter() - start) / iters

        start = time.perf_counter()
        for _ in range(iters):
            run_grouped_fast()
        torch.cuda.synchronize()
        grouped_fast_time = (time.perf_counter() - start) / iters

        start = time.perf_counter()
        for _ in range(iters):
            run_naive()
        torch.cuda.synchronize()
        naive_time = (time.perf_counter() - start) / iters
    print("grouped_time:", grouped_time)
    print("grouped_fast_time:", grouped_fast_time)
    print("eager_time:", naive_time)


@pytest.mark.skipif(os.getenv("NANOMOE_RUN_PERF") != "1", reason="perf tests opt-in")
def test_grouped_mm_backward_efficiency_and_peak_memory() -> None:
    if not hasattr(F, "grouped_mm"):
        pytest.skip("F.grouped_mm is not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(3)
    device = torch.device("cuda")

    num_experts = 128
    num_top_k = 8
    num_tokens = 2048
    hidden_dim = 256
    intermediate_dim = 512

    base_module = ToyMoEKernel(
        num_experts,
        hidden_dim,
        intermediate_dim,
        has_bias=True,
        is_transposed=False,
    ).to(device)

    module_grouped = copy.deepcopy(base_module)
    module_eager = copy.deepcopy(base_module)

    hidden_grouped = torch.randn(num_tokens, hidden_dim, device=device, requires_grad=True)
    hidden_eager = hidden_grouped.detach().clone().requires_grad_(True)

    top_k_index, top_k_weights = _make_topk(num_tokens, num_experts, num_top_k, device=device)

    def run_grouped() -> torch.Tensor:
        out = grouped_mm_experts_forward(module_grouped, hidden_grouped, top_k_index, top_k_weights)
        return out.pow(2).mean()

    def run_grouped_fast() -> torch.Tensor:
        out = grouped_mm_experts_forward_fast(module_grouped, hidden_grouped, top_k_index, top_k_weights)
        return out.pow(2).mean()

    def run_eager() -> torch.Tensor:
        out = _call_eager_or_skip(module_eager, hidden_eager, top_k_index, top_k_weights)
        return out.pow(2).mean()

    def measure(fn, iters: int = 5) -> tuple[float, int | None]:
        total = 0.0
        peak = None
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss = fn()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            total += time.perf_counter() - start
            if device.type == "cuda":
                iter_peak = torch.cuda.max_memory_allocated()
                peak = iter_peak if peak is None else max(peak, iter_peak)
            module_grouped.zero_grad(set_to_none=True)
            module_eager.zero_grad(set_to_none=True)
            hidden_grouped.grad = None
            hidden_eager.grad = None
        return total / iters, peak

    with torch.no_grad():
        run_grouped()
        run_eager()

    module_grouped.zero_grad(set_to_none=True)
    module_eager.zero_grad(set_to_none=True)
    hidden_grouped.grad = None
    hidden_eager.grad = None

    grouped_time, grouped_peak = measure(run_grouped)
    grouped_fast_time, grouped_fast_peak = measure(run_grouped_fast)
    eager_time, eager_peak = measure(run_eager)

    # Print out for visibility
    print("==== MoE grouped_mm forward-backward efficiency ====")
    print("grouped_time:", grouped_time)
    print("grouped_fast_time:", grouped_fast_time)
    print("eager_time:", eager_time)
    assert grouped_peak is not None
    assert grouped_fast_peak is not None
    assert eager_peak is not None
    print("grouped_peak (MB):", grouped_peak / (1024 * 1024))
    print("grouped_fast_peak (MB):", grouped_fast_peak / (1024 * 1024))
    print("eager_peak (MB):", eager_peak / (1024 * 1024))
