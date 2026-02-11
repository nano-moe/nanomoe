"""MoE router abstractions and shared routing utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanomoe.model.config import MoEConfig


def softmax_normalize(router_logits: Tensor) -> Tensor:
    """Normalize router logits with softmax."""
    return F.softmax(router_logits, dim=-1)


def sigmoid_normalize(router_logits: Tensor) -> Tensor:
    """Normalize sigmoid scores to produce a probability simplex."""
    router_probs = torch.sigmoid(router_logits)
    denom = router_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(router_probs.dtype).eps)
    return router_probs / denom


class BaseRouter(nn.Module, ABC):
    """Base class for token-to-expert routers.

    Subclasses implement `compute_router_logits`; shared routing mechanics
    (jitter, normalization, top-k selection, and aux loss) live here.
    """

    def __init__(self, config: MoEConfig, *, prob_normalization: str = "softmax"):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        self.jitter_noise = config.router_jitter_noise
        self.prob_normalization = prob_normalization

        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.num_experts:
            msg = (
                "num_experts_per_tok must satisfy 1 <= num_experts_per_tok <= num_experts, "
                f"got {self.num_experts_per_tok} with {self.num_experts} experts"
            )
            raise ValueError(msg)

    @abstractmethod
    def compute_router_logits(self, hidden_states: Tensor) -> Tensor:
        """Compute per-token, per-expert router logits."""

    def normalize_router_logits(self, router_logits: Tensor) -> Tensor:
        """Normalize logits into routing probabilities."""

    def _select_topk(self, router_probs: Tensor) -> tuple[Tensor, Tensor]:
        """Select top-k experts per token and re-normalize top-k weights."""

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [batch, seq, hidden] or [num_tokens, hidden]

        Returns:
            router_logits: [num_tokens, num_experts]
            expert_indices: [num_tokens, num_experts_per_tok]
            expert_weights: [num_tokens, num_experts_per_tok]
        """

    def compute_aux_loss(self, router_logits: Tensor) -> Tensor:
        """Compute standard load balancing auxiliary loss."""


class LinearRouter(BaseRouter):
    """Base router with a learnable linear gating projection."""

    def __init__(self, config: MoEConfig, *, prob_normalization: str = "softmax"):
        super().__init__(config, prob_normalization=prob_normalization)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def compute_router_logits(self, hidden_states: Tensor) -> Tensor:
        return self.gate(hidden_states)


class NaiveTopKRouter(LinearRouter):
    """Naive top-k router (standard softmax + top-k selection)."""


class StraightThroughTopKRouter(LinearRouter):
    """Top-k hard routing with a straight-through estimator on selected experts."""

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = self._prepare_hidden_states(hidden_states)
        router_logits = self.compute_router_logits(hidden_states)
        router_probs = self.normalize_router_logits(router_logits)

        topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1).indices
        hard_mask = torch.zeros_like(router_probs).scatter(-1, topk_indices, 1.0)
        ste_probs = hard_mask + router_probs - router_probs.detach()

        expert_weights = torch.gather(ste_probs, -1, topk_indices)
        denom = expert_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(expert_weights.dtype).eps)
        expert_weights = expert_weights / denom
        return router_logits, topk_indices, expert_weights


class GumbelStraightThroughTopKRouter(LinearRouter):
    """Top-k router using Gumbel-Softmax perturbation and straight-through selection."""

    def __init__(
        self,
        config: MoEConfig,
        *,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
    ):
        super().__init__(config, prob_normalization="softmax")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if min_temperature <= 0:
            raise ValueError(f"min_temperature must be > 0, got {min_temperature}")
        self.temperature = float(temperature)
        self.min_temperature = float(min_temperature)

    def _sample_gumbel(self, logits: Tensor) -> Tensor:
        u = torch.rand_like(logits).clamp_min(torch.finfo(logits.dtype).eps)
        return -torch.log(-torch.log(u))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = self._prepare_hidden_states(hidden_states)
        router_logits = self.compute_router_logits(hidden_states)
        self._validate_router_logits(router_logits)

        temperature = max(self.temperature, self.min_temperature)
        if self.training:
            sampled_logits = (router_logits + self._sample_gumbel(router_logits)) / temperature
        else:
            sampled_logits = router_logits / temperature

        sampled_probs = F.softmax(sampled_logits, dim=-1)
        topk_indices = torch.topk(sampled_probs, self.num_experts_per_tok, dim=-1).indices
        hard_mask = torch.zeros_like(sampled_probs).scatter(-1, topk_indices, 1.0)
        ste_probs = hard_mask + sampled_probs - sampled_probs.detach()

        expert_weights = torch.gather(ste_probs, -1, topk_indices)
        denom = expert_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(expert_weights.dtype).eps)
        expert_weights = expert_weights / denom
        return router_logits, topk_indices, torch.ones_like(expert_weights) # No need for weighting as we are doing sampling


class PolicyGradientRouter(LinearRouter):
    """Stochastic router using REINFORCE-style policy gradient updates.

    During training, experts are sampled without replacement from the router
    distribution. Call `compute_policy_loss` with token-level rewards to obtain
    a policy gradient loss term for the most recent forward pass.

    WARNING: Not ready for usage
    """

    def __init__(
        self,
        config: MoEConfig,
        *,
        prob_normalization: str = "softmax",
        entropy_coef: float = 0.0,
        baseline_momentum: float = 0.9,
    ):
        import warnings
        warnings.warn("NOT READY FOR USE")
        super().__init__(config, prob_normalization=prob_normalization)
        if entropy_coef < 0:
            raise ValueError(f"entropy_coef must be >= 0, got {entropy_coef}")
        if baseline_momentum < 0 or baseline_momentum >= 1:
            raise ValueError(f"baseline_momentum must be in [0, 1), got {baseline_momentum}")

        self.entropy_coef = entropy_coef
        self.baseline_momentum = baseline_momentum
        self.register_buffer("_reward_baseline", torch.tensor(0.0), persistent=False)
        self._baseline_initialized = False
        self._last_sample_log_probs: Tensor | None = None
        self._last_entropy: Tensor | None = None

    def clear_policy_state(self) -> None:
        """Clear cached policy statistics from the previous sampled forward pass."""
        self._last_sample_log_probs = None
        self._last_entropy = None

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = self._prepare_hidden_states(hidden_states)
        router_logits = self.compute_router_logits(hidden_states)
        self._validate_router_logits(router_logits)
        router_probs = self.normalize_router_logits(router_logits)

        if self.training:
            expert_indices = torch.multinomial(
                router_probs, num_samples=self.num_experts_per_tok, replacement=False
            )
            log_probs = torch.log(router_probs.clamp_min(torch.finfo(router_probs.dtype).eps))
            self._last_sample_log_probs = torch.gather(log_probs, dim=-1, index=expert_indices).sum(dim=-1)
            self._last_entropy = -(router_probs * log_probs).sum(dim=-1)
        else:
            expert_indices, _ = self._select_topk(router_probs)
            self.clear_policy_state()

        expert_weights = torch.gather(router_probs, dim=-1, index=expert_indices)
        denom = expert_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(expert_weights.dtype).eps)
        expert_weights = expert_weights / denom
        return router_logits, expert_indices, expert_weights

    def compute_policy_loss(
        self,
        rewards: Tensor,
        *,
        normalize_rewards: bool = True,
        use_baseline: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """Compute policy gradient loss from cached sampled actions.

        Args:
            rewards: Token-level rewards, shape [num_tokens] or [batch, seq].
            normalize_rewards: If True, normalize reward scale per batch.
            use_baseline: If True, subtract EMA baseline before policy update.
            reduction: Loss reduction mode.
        """
        if self._last_sample_log_probs is None:
            raise RuntimeError("No sampled router actions available. Run a training forward pass first.")

        log_probs = self._last_sample_log_probs
        rewards_flat = rewards.reshape(-1).to(device=log_probs.device, dtype=log_probs.dtype)
        if rewards_flat.numel() != log_probs.numel():
            raise ValueError(
                "rewards must match number of routed tokens. "
                f"Got {rewards_flat.numel()} rewards for {log_probs.numel()} tokens."
            )

        advantages = rewards_flat
        if normalize_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        if use_baseline:
            batch_mean = advantages.mean().detach()
            if self._baseline_initialized:
                self._reward_baseline.mul_(self.baseline_momentum).add_(batch_mean * (1 - self.baseline_momentum))
            else:
                self._reward_baseline.copy_(batch_mean)
                self._baseline_initialized = True
            advantages = advantages - self._reward_baseline

        policy_loss = -(advantages.detach() * log_probs)
        if self.entropy_coef > 0 and self._last_entropy is not None:
            policy_loss = policy_loss - self.entropy_coef * self._last_entropy

        if reduction == "mean":
            return policy_loss.mean()
        if reduction == "sum":
            return policy_loss.sum()
        if reduction == "none":
            return policy_loss
        raise ValueError(f"Unsupported reduction: {reduction}")


SwitchRouter = SwitchTop1Router
StraightThroughRouter = StraightThroughTopKRouter
GumbelSoftmaxStraightThroughRouter = GumbelStraightThroughTopKRouter
