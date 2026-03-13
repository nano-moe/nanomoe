from __future__ import annotations

import pytest
import torch

from nanomoe.lra.model import (
    TransformerClassifierConfig,
    _hull_attention_reference,
    _hull_sparse_attention_reference,
    build_transformer_classifier,
)


def test_transformer_classifier_forward_shape() -> None:
    config = TransformerClassifierConfig(
        vocab_size=32,
        num_classes=10,
        max_seq_len=32,
        input_mode="token",
        pad_token_id=0,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="sdpa",
        pooling="last",
    )
    model = build_transformer_classifier(config)
    inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    attention_mask = inputs.ne(0)

    logits = model(inputs, attention_mask)

    assert logits.shape == (2, 10)


def test_transformer_classifier_supports_cls_pooling() -> None:
    config = TransformerClassifierConfig(
        vocab_size=32,
        num_classes=2,
        max_seq_len=32,
        input_mode="token",
        pad_token_id=0,
        d_model=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="sdpa",
        pooling="cls",
    )
    model = build_transformer_classifier(config)
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)

    logits = model(inputs, attention_mask)

    assert logits.shape == (1, 2)


def test_transformer_classifier_supports_continuous_inputs() -> None:
    config = TransformerClassifierConfig(
        vocab_size=None,
        num_classes=2,
        max_seq_len=32,
        input_mode="continuous",
        input_dim=1,
        d_model=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="sdpa",
        pooling="mean",
    )
    model = build_transformer_classifier(config)
    inputs = torch.randn(2, 8, 1)
    attention_mask = torch.ones((2, 8), dtype=torch.bool)

    logits = model(inputs, attention_mask)

    assert logits.shape == (2, 2)


def test_hull_attention_reference_selects_top_scoring_values() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
    v = torch.tensor([[[[10.0, 1.0], [20.0, 2.0]]]])
    attention_mask = torch.tensor([[True, True]])

    output_temp_1 = _hull_attention_reference(q, k, v, attention_mask, temperature=1.0)
    output_temp_03 = _hull_attention_reference(q, k, v, attention_mask, temperature=0.3)

    assert output_temp_03[0, 0, 0, 0] < output_temp_1[0, 0, 0, 0]
    assert output_temp_03[0, 0, 1, 0] > output_temp_1[0, 0, 1, 0]


def test_transformer_classifier_supports_hullattn_backend() -> None:
    config = TransformerClassifierConfig(
        vocab_size=32,
        num_classes=2,
        max_seq_len=16,
        input_mode="token",
        pad_token_id=0,
        d_model=16,
        num_layers=1,
        num_heads=8,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="hullattn",
        pooling="last",
    )
    model = build_transformer_classifier(config)
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)

    logits = model(inputs, attention_mask)

    assert logits.shape == (1, 2)


def test_transformer_classifier_hullattn_requires_head_dim_2() -> None:
    config = TransformerClassifierConfig(
        vocab_size=32,
        num_classes=2,
        max_seq_len=16,
        input_mode="token",
        pad_token_id=0,
        d_model=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="hullattn",
        pooling="last",
    )
    model = build_transformer_classifier(config)
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)

    with pytest.raises(ValueError, match="head_dim=2"):
        _ = model(inputs, attention_mask)


def test_transformer_classifier_set_hull_temperature_updates_layers() -> None:
    config = TransformerClassifierConfig(
        vocab_size=32,
        num_classes=2,
        max_seq_len=16,
        input_mode="token",
        pad_token_id=0,
        d_model=16,
        num_layers=2,
        num_heads=8,
        ffn_hidden_size=32,
        dropout=0.0,
        attention_backend="hullattn",
        pooling="last",
    )
    model = build_transformer_classifier(config)

    model.set_hull_temperature(0.3)

    assert all(layer.attn.hull_temperature == pytest.approx(0.3) for layer in model.layers)


def test_hull_sparse_attention_reference_topk_all_matches_dense_reference() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
    v = torch.tensor([[[[10.0, 1.0], [20.0, 2.0]]]])
    attention_mask = torch.tensor([[True, True]])

    dense_output = _hull_attention_reference(q, k, v, attention_mask, temperature=0.3)
    sparse_output = _hull_sparse_attention_reference(q, k, v, attention_mask, temperature=0.3, top_k=16)

    torch.testing.assert_close(sparse_output, dense_output)
