from __future__ import annotations

import pytest
import torch

from nanomoe.model.config import MoEConfig
from nanomoe.model.moe_router import (
    GumbelStraightThroughTopKRouter,
    NaiveTopKRouter,
    PolicyGradientRouter,
    StraightThroughTopKRouter,
    SwitchTop1Router,
)


def _router_config() -> MoEConfig:
    return MoEConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
    )


@pytest.mark.parametrize(
    "router_cls",
    [
        NaiveTopKRouter,
        StraightThroughTopKRouter,
        PolicyGradientRouter,
    ],
)
def test_router_forward_shapes_and_weight_norm(router_cls: type[torch.nn.Module]) -> None:
    torch.manual_seed(0)
    config = _router_config()
    router = router_cls(config).train()
    hidden_states = torch.randn(2, 3, config.hidden_size)

    router_logits, expert_indices, expert_weights = router(hidden_states)

    assert router_logits.shape == (6, config.num_experts)
    assert expert_indices.shape == (6, config.num_experts_per_tok)
    assert expert_weights.shape == (6, config.num_experts_per_tok)
    torch.testing.assert_close(expert_weights.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=1e-6)
    assert router.compute_aux_loss(router_logits).shape == torch.Size([])


def test_gumbel_router_forward_shapes_and_weight_norm() -> None:
    torch.manual_seed(0)
    config = _router_config()
    router = GumbelStraightThroughTopKRouter(config, temperature=0.7, min_temperature=0.2).train()
    hidden_states = torch.randn(2, 3, config.hidden_size)

    router_logits, expert_indices, expert_weights = router(hidden_states)

    assert router_logits.shape == (6, config.num_experts)
    assert expert_indices.shape == (6, config.num_experts_per_tok)
    assert expert_weights.shape == (6, config.num_experts_per_tok)
    torch.testing.assert_close(expert_weights.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=1e-6)


def test_switch_router_is_top1() -> None:
    torch.manual_seed(0)
    config = _router_config()
    router = SwitchTop1Router(config).train()
    hidden_states = torch.randn(2, 3, config.hidden_size)

    _, expert_indices, expert_weights = router(hidden_states)
    assert expert_indices.shape == (6, 1)
    assert expert_weights.shape == (6, 1)
    torch.testing.assert_close(expert_weights, torch.ones_like(expert_weights), atol=1e-6, rtol=1e-6)


def test_policy_gradient_router_computes_loss_and_backprop() -> None:
    torch.manual_seed(0)
    config = _router_config()
    router = PolicyGradientRouter(config, entropy_coef=0.01).train()
    hidden_states = torch.randn(2, 3, config.hidden_size)

    router(hidden_states)
    rewards = torch.randn(6)
    loss = router.compute_policy_loss(rewards)

    assert loss.shape == torch.Size([])
    loss.backward()
    assert router.gate.weight.grad is not None


def test_policy_gradient_router_requires_sampled_actions() -> None:
    config = _router_config()
    router = PolicyGradientRouter(config).eval()
    with pytest.raises(RuntimeError):
        _ = router.compute_policy_loss(torch.randn(6))
