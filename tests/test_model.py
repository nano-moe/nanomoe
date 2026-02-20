from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import nanomoe.model.attention as attention_mod
from nanomoe.model.config import MoEConfig
from nanomoe.model.model import MoETransformer, create_model
from nanomoe.model.moe_kernel import MOE_KERNEL_REGISTRY


def _model_config(**kwargs: object) -> MoEConfig:
    return MoEConfig(
        hidden_size=32,
        num_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        **kwargs,
    )


def _unwrap_compiled_attention(model: MoETransformer) -> None:
    for layer in model.layers:
        forward = layer.self_attn.forward
        wrapped = getattr(forward, "__wrapped__", None)
        if wrapped is not None:
            layer.self_attn.forward = wrapped.__get__(layer.self_attn, type(layer.self_attn))


def test_moe_transformer_forward_backward_end_to_end() -> None:
    torch.manual_seed(0)
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    outputs = model(input_ids, use_cache=False)

    assert outputs.logits.shape == (2, 7, config.vocab_size)
    assert isinstance(outputs.aux_loss, torch.Tensor)
    assert outputs.aux_loss.shape == torch.Size([])
    assert outputs.past_key_values is None

    token_loss = F.cross_entropy(
        outputs.logits[:, :-1, :].reshape(-1, config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    (token_loss + outputs.aux_loss).backward()

    assert model.embed_tokens.weight.grad is not None
    router = model.layers[0].mlp.router
    assert router.gate.weight.grad is not None


@torch.no_grad()
def test_moe_transformer_kv_cache_matches_full_forward_last_token() -> None:
    torch.manual_seed(1)
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    full_outputs = model(input_ids, use_cache=False)
    prefix_outputs = model(input_ids[:, :-1], use_cache=True)
    step_outputs = model(input_ids[:, -1:], past_key_values=prefix_outputs.past_key_values, use_cache=True)

    torch.testing.assert_close(
        step_outputs.logits[:, -1, :],
        full_outputs.logits[:, -1, :],
        atol=1e-6,
        rtol=1e-5,
    )
    assert prefix_outputs.past_key_values is not None
    assert len(prefix_outputs.past_key_values) == config.num_layers
    assert step_outputs.past_key_values is not None
    assert step_outputs.past_key_values[0] is not None
    assert step_outputs.past_key_values[0][0].shape[2] == input_ids.shape[1]


@torch.no_grad()
def test_moe_transformer_generate_grows_sequence() -> None:
    torch.manual_seed(2)
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    generated = model.generate(input_ids, max_new_tokens=3, temperature=0.0, eos_token_id=-1)

    assert generated.shape == (2, 7)
    torch.testing.assert_close(generated[:, :4], input_ids)


def test_moe_transformer_validates_past_key_value_length() -> None:
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)

    with pytest.raises(ValueError, match="one entry per layer"):
        _ = model(torch.randint(0, config.vocab_size, (1, 1)), past_key_values=[None], use_cache=True)


def test_create_model_uses_default_router_type() -> None:
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)

    assert model.layers[0].mlp.router.__class__.__name__ == "NaiveTopKRouter"


def test_create_model_uses_default_attention_type_and_auto_kernel() -> None:
    config = _model_config()
    model = create_model(config)
    _unwrap_compiled_attention(model)

    assert model.layers[0].self_attn.attention_fn.__name__ == "_fsdp_attention"
    expected_kernel = "grouped_mm" if hasattr(F, "grouped_mm") else "eager_mm"
    assert model.layers[0].mlp.moe_kernel == expected_kernel
    assert model.layers[0].mlp.experts.kernel_fn is MOE_KERNEL_REGISTRY[expected_kernel]


def test_create_model_honors_attention_type_and_moe_kernel_overrides() -> None:
    config = _model_config(attention_type="fsdp_attention", moe_kernel="eager_mm")
    model = create_model(config)
    _unwrap_compiled_attention(model)

    assert model.layers[0].self_attn.attention_fn.__name__ == "_fsdp_attention"
    assert model.layers[0].mlp.moe_kernel == "eager_mm"
    assert model.layers[0].mlp.experts.kernel_fn is MOE_KERNEL_REGISTRY["eager_mm"]


def test_config_round_trip_includes_attention_type_and_moe_kernel() -> None:
    config = _model_config(attention_type="flex_attention", moe_kernel="grouped_mm_fast")

    loaded = MoEConfig.from_dict(config.to_dict())

    assert loaded.attention_type == "flex_attention"
    assert loaded.moe_kernel == "grouped_mm_fast"


def test_config_rejects_unknown_attention_type() -> None:
    with pytest.raises(ValueError, match="Unsupported attention_type"):
        _model_config(attention_type="bad_attention")


def test_config_rejects_unknown_moe_kernel() -> None:
    with pytest.raises(ValueError, match="Unsupported moe_kernel"):
        _model_config(moe_kernel="bad_kernel")


@torch.compiler.disable
def test_moe_transformer_flex_attention_defaults_single_doc_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device):
        captured["B"] = B
        captured["H"] = H
        captured["Q_LEN"] = Q_LEN
        captured["KV_LEN"] = KV_LEN
        captured["device"] = device

        # Defaults should represent one document per batch element with length=seq_len.
        assert bool(mask_mod(0, 0, 1, 0))
        assert not bool(mask_mod(0, 0, 1, 2))
        assert bool(mask_mod(1, 0, 4, 0))
        assert bool(mask_mod(1, 0, 4, 4))
        assert not bool(mask_mod(1, 0, 0, 4))
        return "fake_block_mask"

    def fake_flex_attention(q, k, v, block_mask):
        captured["block_mask"] = block_mask
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_mod, "create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(attention_mod, "flex_attention", fake_flex_attention)

    config = _model_config(attention_type="flex_attention")
    model = create_model(config)
    _unwrap_compiled_attention(model)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    outputs = model(input_ids, use_cache=False)

    assert outputs.logits.shape == (2, 5, config.vocab_size)
    assert captured["B"] == 2
    assert captured["H"] is None
    assert captured["Q_LEN"] == 5
    assert captured["KV_LEN"] == 5
    assert captured["block_mask"] == "fake_block_mask"
