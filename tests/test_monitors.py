from __future__ import annotations

from typing import cast

import torch

from nanomoe.data.types import PackedBatch
from nanomoe.model.config import MoEConfig
from nanomoe.model.model import MoETransformer, TransformerBlock, create_model
from nanomoe.monitors import attention_logit_norms, hidden_state_cosine_similarities


def _model_config() -> MoEConfig:
    return MoEConfig(
        hidden_size=16,
        num_layers=3,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        moe_kernel="eager_mm",
    )


def _unwrap_compiled_attention(model: MoETransformer) -> None:
    for layer in model.layers:
        tb = cast(TransformerBlock, layer)
        forward = tb.self_attn.forward
        wrapped = getattr(forward, "__wrapped__", None)
        if wrapped is not None:
            tb.self_attn.forward = wrapped.__get__(tb.self_attn, type(tb.self_attn))


def test_attention_logit_norms_returns_per_layer_stats_and_restores_mode() -> None:
    torch.manual_seed(0)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    result = attention_logit_norms(model, input_ids)

    assert model.training
    assert len(result.layers) == model.config.num_layers
    for layer_idx, stats in enumerate(result.layers):
        assert stats.layer_idx == layer_idx
        assert stats.num_batches == 1
        assert stats.num_logits > 0
        assert stats.mean_query_l2_norm >= 0.0
        assert stats.rms_logit >= 0.0
        assert stats.mean_abs_logit >= 0.0
        assert stats.max_abs_logit >= 0.0


def test_attention_logit_norms_accepts_dict_batches_with_packing_mask() -> None:
    torch.manual_seed(1)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    dataset = [
        {
            "input_ids": input_ids,
            "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
            "packing_doc_ids": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "packing_seq_lens": torch.tensor([6]),
        }
    ]

    result = attention_logit_norms(model, dataset)

    assert len(result.layers) == model.config.num_layers
    assert all(stats.num_batches == 1 for stats in result.layers)


def test_hidden_state_cosine_similarities_returns_layer_and_neighbour_stats() -> None:
    torch.manual_seed(2)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    result = hidden_state_cosine_similarities(model, input_ids)

    assert len(result.intra_sequence) == model.config.num_layers
    assert len(result.neighbouring_layers) == model.config.num_layers - 1
    assert len(result.first_layer_reference) == model.config.num_layers - 1
    assert len(result.router_usage_entropy) == model.config.num_layers
    assert len(result.router_logits) == model.config.num_layers
    for layer_idx, stats in enumerate(result.intra_sequence):
        assert stats.layer_idx == layer_idx
        assert stats.num_sequences == input_ids.shape[0]
        assert -1.0 <= stats.mean_off_diagonal_cosine <= 1.0
    for layer_idx, stats in enumerate(result.neighbouring_layers):
        assert stats.layer_idx == layer_idx
        assert stats.next_layer_idx == layer_idx + 1
        assert stats.num_tokens == input_ids.numel()
        assert -1.0 <= stats.mean_token_cosine <= 1.0
        assert stats.mean_rms_distance >= 0.0
        assert torch.isfinite(torch.tensor(stats.mean_relative_rms_change))
    assert [(stats.layer_idx, stats.next_layer_idx) for stats in result.first_layer_reference] == [(0, 1), (0, 2)]
    for stats in result.first_layer_reference:
        assert stats.num_tokens == input_ids.numel()
        assert -1.0 <= stats.mean_token_cosine <= 1.0
        assert stats.mean_rms_distance >= 0.0
        assert torch.isfinite(torch.tensor(stats.mean_relative_rms_change))
    for layer_idx, stats in enumerate(result.router_usage_entropy):
        assert stats.layer_idx == layer_idx
        assert stats.num_tokens == input_ids.numel()
        assert stats.num_assignments == input_ids.numel() * model.config.num_experts_per_tok
        assert 0.0 <= stats.entropy <= torch.log2(torch.tensor(float(model.config.num_experts))).item()
        assert result.router_logits[layer_idx].shape == (input_ids.numel(), model.config.num_experts)


def test_hidden_state_cosine_similarities_flag_keeps_both_layer_comparison_groups() -> None:
    torch.manual_seed(22)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    result = hidden_state_cosine_similarities(model, input_ids, first_layer_as_reference=True)

    assert len(result.neighbouring_layers) == model.config.num_layers - 1
    assert [(stats.layer_idx, stats.next_layer_idx) for stats in result.neighbouring_layers] == [(0, 1), (1, 2)]
    assert [(stats.layer_idx, stats.next_layer_idx) for stats in result.first_layer_reference] == [(0, 1), (0, 2)]
    assert all(stats.num_tokens == input_ids.numel() for stats in result.neighbouring_layers)
    assert all(stats.num_tokens == input_ids.numel() for stats in result.first_layer_reference)


def test_hidden_state_cosine_similarities_uses_2d_attention_mask_for_valid_tokens() -> None:
    torch.manual_seed(3)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    batch = {
        "input_ids": torch.randint(0, model.config.vocab_size, (1, 4)),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }

    result = hidden_state_cosine_similarities(model, [batch])

    assert all(stats.num_sequences == 1 for stats in result.intra_sequence)
    assert all(stats.num_tokens == 2 for stats in result.neighbouring_layers)
    assert all(stats.num_tokens == 2 for stats in result.first_layer_reference)
    assert all(stats.mean_rms_distance >= 0.0 for stats in result.neighbouring_layers)
    assert all(stats.mean_rms_distance >= 0.0 for stats in result.first_layer_reference)


def test_hidden_state_cosine_similarities_accepts_packed_batches() -> None:
    torch.manual_seed(4)
    model = create_model(_model_config())
    _unwrap_compiled_attention(model)
    batch = PackedBatch(
        tokens=torch.randint(0, model.config.vocab_size, (5,)),
        position_ids=torch.tensor([0, 1, 2, 0, 1]),
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        token_weights=torch.ones(5),
        labels=torch.randint(0, model.config.vocab_size, (5,)),
    )

    result = hidden_state_cosine_similarities(model, [batch])

    assert all(stats.num_sequences == 2 for stats in result.intra_sequence)
    assert all(stats.num_tokens == 5 for stats in result.neighbouring_layers)
    assert all(stats.num_tokens == 5 for stats in result.first_layer_reference)
    assert all(stats.mean_rms_distance >= 0.0 for stats in result.neighbouring_layers)
    assert all(stats.mean_rms_distance >= 0.0 for stats in result.first_layer_reference)
    assert all(stats.num_tokens == 5 for stats in result.router_usage_entropy)
    assert all(stats.num_assignments == 5 for stats in result.router_usage_entropy)
    assert all(layer_logits.shape == (5, model.config.num_experts) for layer_logits in result.router_logits)
