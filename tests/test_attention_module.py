from __future__ import annotations

import importlib.util
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _load_attention_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "nanomoe" / "model" / "attention.py"
    spec = importlib.util.spec_from_file_location("nanomoe_attention_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load attention module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_mod = _load_attention_module()


def _unwrap_compiled_forward(module: torch.nn.Module) -> None:
    forward = module.forward
    wrapped = getattr(forward, "__wrapped__", None)
    if wrapped is not None:
        module.forward = wrapped.__get__(module, type(module))


def _doc_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    doc_ids = torch.empty(total_tokens, dtype=torch.long, device=cu_seqlens.device)
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        doc_ids[start:end] = i
    return doc_ids


def _position_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    pos = torch.empty(total_tokens, dtype=torch.long, device=cu_seqlens.device)
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        pos[start:end] = torch.arange(end - start, device=cu_seqlens.device, dtype=torch.long)
    return pos


def _packed_4d_mask_from_cu_seqlens(cu_seqlens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    doc_ids = torch.empty(total_tokens, dtype=torch.long, device=cu_seqlens.device)
    for i in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        doc_ids[start:end] = i

    q_idx = torch.arange(total_tokens, device=cu_seqlens.device).unsqueeze(1)
    kv_idx = torch.arange(total_tokens, device=cu_seqlens.device).unsqueeze(0)
    allowed = (q_idx >= kv_idx) & (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))

    mask = torch.zeros(total_tokens, total_tokens, dtype=dtype, device=cu_seqlens.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def _flash_style_attention_with_cu_seqlens(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
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


def _benchmark_cuda_ms_and_peak_bytes(fn, warmup: int, iters: int) -> tuple[float, int]:
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
    return total_s * 1000.0 / iters, peak_bytes


def test_fsdp_attention_packed_4d_mask_matches_segment_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    packed_mask_4d = _packed_4d_mask_from_cu_seqlens(cu_seqlens, dtype=q.dtype)

    out_fsdp = attention_mod._fsdp_attention(
        q=q,
        k=k,
        v=v,
        attn_mask=packed_mask_4d,
        dropout_p=0.0,
        is_causal=False,
        position_ids=None,
        packing_doc_ids=None,
        packing_seq_lens=None,
    )
    out_ref = _flash_style_attention_with_cu_seqlens(q, k, v, cu_seqlens)

    torch.testing.assert_close(out_fsdp, out_ref, atol=1e-6, rtol=1e-5)


@torch.compiler.disable
def test_attention_forward_flex_uses_packing_doc_ids_and_seq_lens(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    captured: dict[str, object] = {}

    def fake_create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device):
        captured["B"] = B
        captured["H"] = H
        captured["Q_LEN"] = Q_LEN
        captured["KV_LEN"] = KV_LEN
        captured["device"] = device

        # q=1,kv=0 is causal and same doc.
        assert bool(mask_mod(0, 0, 1, 0))
        # q=1,kv=2 is not causal.
        assert not bool(mask_mod(0, 0, 1, 2))
        # q=3 (doc=1), kv=1 (doc=0) crosses docs.
        assert not bool(mask_mod(0, 0, 3, 1))
        # q=5 exceeds provided valid length.
        assert not bool(mask_mod(0, 0, 5, 0))
        # kv=5 exceeds provided valid length.
        assert not bool(mask_mod(0, 0, 4, 5))
        return "fake_block_mask"

    def fake_flex_attention(q, k, v, block_mask):
        captured["block_mask"] = block_mask
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_mod, "create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(attention_mod, "flex_attention", fake_flex_attention)

    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10000.0,
        attention_dropout=0.0,
        attention_type="flex_attention",
    )
    attn = attention_mod.Attention(config)
    _unwrap_compiled_forward(attn)

    hidden_states = torch.randn(1, 6, config.hidden_size)
    position_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)
    packing_doc_ids = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
    packing_seq_lens = torch.tensor([5], dtype=torch.long)

    out, _ = attn(
        hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        packing_doc_ids=packing_doc_ids,
        packing_seq_lens=packing_seq_lens,
    )

    assert out.shape == hidden_states.shape
    assert captured["B"] == 1
    assert captured["H"] is None
    assert captured["Q_LEN"] == 6
    assert captured["KV_LEN"] == 6
    assert captured["block_mask"] == "fake_block_mask"


def test_flex_attention_requires_packing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    monkeypatch.setattr(attention_mod, "create_block_mask", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(attention_mod, "flex_attention", lambda q, k, v, block_mask: torch.zeros_like(q))

    with pytest.raises(ValueError, match="requires packing_doc_ids and packing_seq_lens"):
        attention_mod._flex_attention(
            q=q,
            k=k,
            v=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            position_ids=None,
            packing_doc_ids=None,
            packing_seq_lens=None,
        )


@torch.compiler.disable
def test_attention_forward_flex_matches_fsdp_for_packed_input() -> None:
    if attention_mod.create_block_mask is None or attention_mod.flex_attention is None:
        pytest.skip("torch.nn.attention.flex_attention is not available")

    torch.manual_seed(11)
    device = torch.device("cpu")
    cu_seqlens = torch.tensor([0, 4, 7, 9], dtype=torch.int32, device=device)
    total_tokens = int(cu_seqlens[-1].item())

    position_ids = _position_ids_from_cu_seqlens(cu_seqlens).unsqueeze(0)
    packing_doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens).unsqueeze(0)
    packing_seq_lens = torch.tensor([total_tokens], dtype=torch.long, device=device)
    packed_mask_4d = _packed_4d_mask_from_cu_seqlens(cu_seqlens, dtype=torch.float32)

    common_cfg = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )
    fsdp_cfg = SimpleNamespace(**common_cfg, attention_type="fsdp_attention")
    flex_cfg = SimpleNamespace(**common_cfg, attention_type="flex_attention")
    attn_fsdp = attention_mod.Attention(fsdp_cfg).eval()
    attn_flex = attention_mod.Attention(flex_cfg).eval()
    _unwrap_compiled_forward(attn_fsdp)
    _unwrap_compiled_forward(attn_flex)
    attn_flex.load_state_dict(attn_fsdp.state_dict())

    hidden_states = torch.randn(1, total_tokens, fsdp_cfg.hidden_size, device=device)

    out_fsdp, _ = attn_fsdp(
        hidden_states,
        attention_mask=packed_mask_4d,
        position_ids=position_ids,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
        out_flex, _ = attn_flex(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            packing_doc_ids=packing_doc_ids,
            packing_seq_lens=packing_seq_lens,
        )

    torch.testing.assert_close(out_flex, out_fsdp, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(os.getenv("NANOMOE_RUN_PERF") != "1", reason="perf tests opt-in")
@torch.compiler.disable
def test_attention_forward_flex_vs_fsdp_efficiency_cuda() -> None:
    if attention_mod.create_block_mask is None or attention_mod.flex_attention is None:
        pytest.skip("torch.nn.attention.flex_attention is not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cu_seqlens = torch.tensor([0, 256, 480, 800, 1024, 4096, 9000, 20000], dtype=torch.int32, device=device)
    total_tokens = int(cu_seqlens[-1].item())

    position_ids = _position_ids_from_cu_seqlens(cu_seqlens).unsqueeze(0)
    packing_doc_ids = _doc_ids_from_cu_seqlens(cu_seqlens).unsqueeze(0)
    packing_seq_lens = torch.tensor([total_tokens], dtype=torch.long, device=device)
    packed_mask_4d = _packed_4d_mask_from_cu_seqlens(cu_seqlens, dtype=dtype)

    common_cfg = dict(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )
    fsdp_cfg = SimpleNamespace(**common_cfg, attention_type="fsdp_attention")
    flex_cfg = SimpleNamespace(**common_cfg, attention_type="flex_attention")
    attn_fsdp = attention_mod.Attention(fsdp_cfg).to(device=device, dtype=dtype).eval()
    attn_flex = attention_mod.Attention(flex_cfg).to(device=device, dtype=dtype).eval()
    # _unwrap_compiled_forward(attn_fsdp)
    # _unwrap_compiled_forward(attn_flex)
    attn_flex.load_state_dict(attn_fsdp.state_dict())

    hidden_states = torch.randn(1, total_tokens, fsdp_cfg.hidden_size, device=device, dtype=dtype)

    def run_fsdp() -> torch.Tensor:
        out, _ = attn_fsdp(
            hidden_states,
            attention_mask=packed_mask_4d,
            position_ids=position_ids,
        )
        return out

    def run_flex() -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
            out, _ = attn_flex(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                packing_doc_ids=packing_doc_ids,
                packing_seq_lens=packing_seq_lens,
            )
        return out

    out_fsdp = run_fsdp()
    out_flex = run_flex()
    torch.testing.assert_close(out_flex, out_fsdp, atol=5e-2, rtol=5e-2)

    warmup, iters = 5, 20
    fsdp_ms, fsdp_peak = _benchmark_cuda_ms_and_peak_bytes(run_fsdp, warmup=warmup, iters=iters)
    flex_ms, flex_peak = _benchmark_cuda_ms_and_peak_bytes(run_flex, warmup=warmup, iters=iters)
    speed_ratio = fsdp_ms / max(flex_ms, 1e-9)
    memory_ratio = fsdp_peak / max(flex_peak, 1)

    print(f"attention module packed benchmark ({dtype}, tokens={total_tokens}, hidden={fsdp_cfg.hidden_size})")
    print(f"fsdp(4d mask): {fsdp_ms:.3f} ms/iter, peak={fsdp_peak / (1024 * 1024):.1f} MB")
    print(f"flex(doc mask): {flex_ms:.3f} ms/iter, peak={flex_peak / (1024 * 1024):.1f} MB")
    print(f"fsdp/flex ratio: {speed_ratio:.3f}x")
    print(f"fsdp/flex peak memory ratio: {memory_ratio:.3f}x")

    assert fsdp_ms > 0.0
    assert flex_ms > 0.0
    assert fsdp_peak > 0
    assert flex_peak > 0
