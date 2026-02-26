"""Tests for packed-sequence attention masking equivalence.

These tests check that three packing-aware masking styles are equivalent:
- Flash-attention style varlen execution with ``cu_seqlens`` (reference loop per segment)
- Flex attention with a block mask
- A dense 4D additive attention mask
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _doc_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    doc_ids = torch.empty(total_tokens, dtype=torch.long, device=cu_seqlens.device)
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        doc_ids[start:end] = i
    return doc_ids


def _packed_4d_mask_from_cu_seqlens(cu_seqlens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    device = cu_seqlens.device

    doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens)
    q_idx = torch.arange(total_tokens, device=device).unsqueeze(1)
    kv_idx = torch.arange(total_tokens, device=device).unsqueeze(0)

    allowed = (q_idx >= kv_idx) & (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))

    mask = torch.zeros(total_tokens, total_tokens, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def _flash_style_attention_with_cu_seqlens(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reference varlen attention: run causal SDPA independently per packed segment."""
    out = torch.zeros_like(q)
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        out[:, :, start:end, :] = F.scaled_dot_product_attention(
            q[:, :, start:end, :],
            k[:, :, start:end, :],
            v[:, :, start:end, :],
            dropout_p=0.0,
            is_causal=True,
        )
    return out


def _attention_with_4d_mask(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    mask = _packed_4d_mask_from_cu_seqlens(cu_seqlens, dtype=q.dtype)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )


def _attention_with_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens)
    q_idx = torch.arange(total_tokens, device=cu_seqlens.device).unsqueeze(1)
    kv_idx = torch.arange(total_tokens, device=cu_seqlens.device).unsqueeze(0)
    allowed = (q_idx >= kv_idx) & (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))

    def mask_mod(batch, head, q_index, kv_index):
        del batch, head
        return allowed[q_index, kv_index]

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=q.device,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
        return cast(torch.Tensor, flex_attention(q, k, v, block_mask=block_mask))


def _benchmark_cuda_ms_and_peak_bytes(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    total_s = 0.0
    peak_bytes = 0
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        total_s += time.perf_counter() - start
        peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
    elapsed_ms = total_s * 1000.0 / iters
    return elapsed_ms, peak_bytes


def _make_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 5, 9, 13], dtype=torch.int32, device=device)
    bsz, nheads, total_tokens, head_dim = 16, 4, int(cu_seqlens[-1].item()), 16

    q = torch.randn(bsz, nheads, total_tokens, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(bsz, nheads, total_tokens, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(bsz, nheads, total_tokens, head_dim, device=device, dtype=torch.float32)
    return q, k, v, cu_seqlens


def test_packed_attention_cu_seqlens_matches_4d_mask_forward_backward() -> None:
    q, k, v, cu_seqlens = _make_qkv(device=torch.device("cpu"))

    out_cu = _flash_style_attention_with_cu_seqlens(q, k, v, cu_seqlens)
    out_4d = _attention_with_4d_mask(q, k, v, cu_seqlens)
    torch.testing.assert_close(out_cu, out_4d, atol=1e-6, rtol=1e-5)

    grad_out = torch.randn_like(out_cu)

    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    (_flash_style_attention_with_cu_seqlens(q1, k1, v1, cu_seqlens) * grad_out).sum().backward()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    (_attention_with_4d_mask(q2, k2, v2, cu_seqlens) * grad_out).sum().backward()

    torch.testing.assert_close(q1.grad, q2.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(k1.grad, k2.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(v1.grad, v2.grad, atol=1e-6, rtol=1e-5)


def test_packed_attention_cu_seqlens_matches_flex_attention_forward() -> None:
    q, k, v, cu_seqlens = _make_qkv(device=torch.device("cpu"))

    out_cu = _flash_style_attention_with_cu_seqlens(q, k, v, cu_seqlens)
    out_flex = _attention_with_flex(q, k, v, cu_seqlens)

    torch.testing.assert_close(out_cu, out_flex, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(os.getenv("NANOMOE_RUN_PERF") != "1", reason="perf tests opt-in")
def test_packed_attention_runtime_benchmark_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 8 documents packed into one 2K-token sequence.
    lengths = [256, 224, 320, 192, 384, 160, 256, 256, 1024, 4096]
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lengths), dim=0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_tokens = int(cu_seqlens[-1].item())

    q = torch.randn(4, 16, total_tokens, 64, device=device, dtype=dtype)
    k = torch.randn(4, 16, total_tokens, 64, device=device, dtype=dtype)
    v = torch.randn(4, 16, total_tokens, 64, device=device, dtype=dtype)

    mask_4d = _packed_4d_mask_from_cu_seqlens(cu_seqlens, dtype=q.dtype)
    doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens)
    q_idx = torch.arange(total_tokens, device=device).unsqueeze(1)
    kv_idx = torch.arange(total_tokens, device=device).unsqueeze(0)
    allowed = (q_idx >= kv_idx) & (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))

    def mask_mod(batch, head, q_index, kv_index):
        del batch, head
        return allowed[q_index, kv_index]

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
    )

    def run_cu() -> torch.Tensor:
        return _flash_style_attention_with_cu_seqlens(q, k, v, cu_seqlens)

    def run_4d() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_4d,
            dropout_p=0.0,
            is_causal=False,
        )

    def run_flex() -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
            return cast(torch.Tensor, flex_attention(q, k, v, block_mask=block_mask))

    run_flex_compiled = None
    flex_compile_error = None
    run_4d_compiled = None
    mask4d_compile_error = None
    if hasattr(torch, "compile"):
        try:
            compiled_flex_attention = torch.compile(flex_attention)

            def run_flex_compiled() -> torch.Tensor:
                return cast(torch.Tensor, compiled_flex_attention(q, k, v, block_mask=block_mask))
        except Exception as exc:  # pragma: no cover - env/backend dependent
            flex_compile_error = exc
        try:

            def _attention_with_4d_mask_compiled(
                q_in: torch.Tensor,
                k_in: torch.Tensor,
                v_in: torch.Tensor,
                mask_in: torch.Tensor,
            ) -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    q_in,
                    k_in,
                    v_in,
                    attn_mask=mask_in,
                    dropout_p=0.0,
                    is_causal=False,
                )

            compiled_4d_attention = torch.compile(_attention_with_4d_mask_compiled)

            def run_4d_compiled() -> torch.Tensor:
                return compiled_4d_attention(q, k, v, mask_4d)
        except Exception as exc:  # pragma: no cover - env/backend dependent
            mask4d_compile_error = exc

    # Sanity check correctness at benchmark shape.
    out_cu = run_cu()
    out_4d = run_4d()
    out_flex = run_flex()
    torch.testing.assert_close(out_cu, out_4d, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_cu, out_flex, atol=5e-2, rtol=5e-2)
    if run_flex_compiled is not None:
        out_flex_compiled = run_flex_compiled()
        torch.testing.assert_close(out_cu, out_flex_compiled, atol=5e-2, rtol=5e-2)
    if run_4d_compiled is not None:
        out_4d_compiled = run_4d_compiled()
        torch.testing.assert_close(out_cu, out_4d_compiled, atol=5e-2, rtol=5e-2)

    warmup, iters = 5, 30
    cu_ms, cu_peak = _benchmark_cuda_ms_and_peak_bytes(run_cu, warmup=warmup, iters=iters)
    flex_ms, flex_peak = _benchmark_cuda_ms_and_peak_bytes(run_flex, warmup=warmup, iters=iters)
    mask4d_ms, mask4d_peak = _benchmark_cuda_ms_and_peak_bytes(run_4d, warmup=warmup, iters=iters)
    flex_compiled_ms = None
    flex_compiled_peak = None
    mask4d_compiled_ms = None
    mask4d_compiled_peak = None
    if run_flex_compiled is not None:
        flex_compiled_ms, flex_compiled_peak = _benchmark_cuda_ms_and_peak_bytes(
            run_flex_compiled, warmup=warmup, iters=iters
        )
    if run_4d_compiled is not None:
        mask4d_compiled_ms, mask4d_compiled_peak = _benchmark_cuda_ms_and_peak_bytes(
            run_4d_compiled, warmup=warmup, iters=iters
        )

    print(f"packed-attn benchmark ({dtype}, tokens={total_tokens}, heads=16, head_dim=64)")
    print(f"cu_seqlens varlen SDPA: {cu_ms:.3f} ms/iter, peak={cu_peak / (1024 * 1024):.1f} MB")
    print(f"flex attention       : {flex_ms:.3f} ms/iter, peak={flex_peak / (1024 * 1024):.1f} MB")
    if flex_compiled_ms is not None:
        assert flex_compiled_peak is not None
        print(
            f"flex attention+compile: {flex_compiled_ms:.3f} ms/iter, peak={flex_compiled_peak / (1024 * 1024):.1f} MB"
        )
    elif flex_compile_error is not None:
        print(f"flex attention+compile: unavailable ({type(flex_compile_error).__name__}: {flex_compile_error})")
    else:
        print("flex attention+compile: unavailable (torch.compile not present)")
    print(f"4d attention mask    : {mask4d_ms:.3f} ms/iter, peak={mask4d_peak / (1024 * 1024):.1f} MB")
    if mask4d_compiled_ms is not None:
        assert mask4d_compiled_peak is not None
        print(
            "4d mask+compile      :"
            f" {mask4d_compiled_ms:.3f} ms/iter, peak={mask4d_compiled_peak / (1024 * 1024):.1f} MB"
        )
    elif mask4d_compile_error is not None:
        print(f"4d mask+compile      : unavailable ({type(mask4d_compile_error).__name__}: {mask4d_compile_error})")
    else:
        print("4d mask+compile      : unavailable (torch.compile not present)")

    assert cu_ms > 0.0
    assert flex_ms > 0.0
    assert mask4d_ms > 0.0
    assert cu_peak > 0
    assert flex_peak > 0
    assert mask4d_peak > 0
    if flex_compiled_ms is not None:
        assert flex_compiled_ms > 0.0
        assert flex_compiled_peak is not None
        assert flex_compiled_peak > 0
    if mask4d_compiled_ms is not None:
        assert mask4d_compiled_ms > 0.0
        assert mask4d_compiled_peak is not None
        assert mask4d_compiled_peak > 0
