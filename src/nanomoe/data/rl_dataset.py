"""RL dataset that pulls sampler groups, computes advantages, and packs batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from nanomoe.data.packing import pack_sequences
from nanomoe.data.types import PackedBatch, Sample


class Sampler(Protocol):
    def sample_and_score(self, prompt: Tensor) -> ScoredGroup: ...


@dataclass
class ScoredGroup:
    tokens: Tensor  # [N, seq_len]
    log_probs: Tensor  # [N, gen_len]
    rewards: Tensor  # [N]
    prompt_len: int


class RLDatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    seq_len: int = 4096  # Total tokens per packed sequence (not per document — see max_seq_len)
    max_tokens_per_batch: int | None = None
    max_attempts: int = 10_000
    reward_std_eps: float = 1e-6
    reward_clip: float | None = None
    advantage_clip: float | None = None
    pad_token_id: int = 0


class RLDatasetStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    groups_seen: int = 0
    valid_groups: int = 0
    zero_var_groups: int = 0
    batches_emitted: int = 0
    attempts: int = 0

    def metrics(self) -> dict[str, float]:
        attempts_per_batch = self.attempts / max(self.batches_emitted, 1)
        return {
            "zero_var_groups": float(self.zero_var_groups),
            "valid_groups": float(self.valid_groups),
            "attempts_per_batch": float(attempts_per_batch),
        }


class RLDataset:
    """Dataset that yields PackedBatch objects from a sampler and prompt source."""

    def __init__(
        self,
        prompt_source: Iterable[Tensor],
        sampler: Sampler,
        config: RLDatasetConfig | None = None,
    ):
        self.prompt_source = prompt_source
        self.sampler = sampler
        self.config = config or RLDatasetConfig()
        self._stats = RLDatasetStats()
        self._group_id = 0

    def _group_to_samples(self, group: ScoredGroup) -> list[Sample]:
        tokens = group.tokens
        log_probs = group.log_probs
        rewards = group.rewards

        if tokens.ndim != 2:
            raise ValueError("ScoredGroup.tokens must be 2D [N, seq_len]")
        if rewards.ndim != 1 or rewards.shape[0] != tokens.shape[0]:
            raise ValueError("ScoredGroup.rewards must be shape [N]")
        if log_probs.ndim != 2 or log_probs.shape[0] != tokens.shape[0]:
            raise ValueError("ScoredGroup.log_probs must be shape [N, gen_len]")

        num_samples = tokens.shape[0]
        samples: list[Sample] = []

        for i in range(num_samples):
            seq = tokens[i].tolist()
            lp = log_probs[i].tolist() if log_probs.numel() > 0 else []
            prompt_len = group.prompt_len

            if prompt_len > len(seq):
                raise ValueError("prompt_len exceeds sequence length")

            loss_mask = [0] * prompt_len + [1] * (len(seq) - prompt_len)

            # Trim trailing pad tokens (assumes pad_token_id).
            while seq and seq[-1] == self.config.pad_token_id:
                seq.pop()
                loss_mask.pop()
                if lp:
                    lp.pop()

            samples.append(
                Sample(
                    tokens=seq,
                    loss_mask=loss_mask,
                    log_probs=lp,
                    reward=float(rewards[i].item()),
                    prompt_len=prompt_len,
                    group_id=self._group_id,
                )
            )

        self._group_id += 1
        return samples

    def _apply_advantages(self, samples: list[Sample], rewards: Tensor | None = None) -> None:
        if rewards is None:
            rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
            if self.config.reward_clip is not None:
                rewards = rewards.clamp(-self.config.reward_clip, self.config.reward_clip)

        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        if self.config.advantage_clip is not None:
            advantages = advantages.clamp(-self.config.advantage_clip, self.config.advantage_clip)

        for sample, adv in zip(samples, advantages, strict=True):
            weights = [adv.item() if m else 0.0 for m in sample.loss_mask[1:]]
            weights.append(0.0)
            sample.token_weights = weights

    def __iter__(self) -> Iterator[PackedBatch]:
        prompt_iter = iter(self.prompt_source)
        buffer: list[Sample] = []
        buffer_tokens = 0
        skipped = 0
        pack_limit = self.config.max_tokens_per_batch or self.config.seq_len

        while True:
            try:
                prompt = next(prompt_iter)
            except StopIteration:
                if buffer:
                    for packed in pack_sequences(buffer, max_tokens_per_batch=pack_limit):
                        self._stats.batches_emitted += 1
                        yield packed
                return

            self._stats.attempts += 1
            group = self.sampler.sample_and_score(prompt)
            self._stats.groups_seen += 1

            rewards = group.rewards.float()
            if self.config.reward_clip is not None:
                rewards = rewards.clamp(-self.config.reward_clip, self.config.reward_clip)
            if rewards.numel() <= 1:
                reward_std = 0.0
            else:
                reward_std = rewards.std().item()

            if reward_std <= self.config.reward_std_eps:
                self._stats.zero_var_groups += 1
                skipped += 1
                if skipped >= self.config.max_attempts:
                    raise RuntimeError("Too many skipped groups (zero reward variance).")
                continue

            skipped = 0
            self._stats.valid_groups += 1

            samples = self._group_to_samples(group)
            self._apply_advantages(samples, rewards=rewards)

            buffer.extend(samples)
            buffer_tokens += sum(len(s.tokens) for s in samples)

            if buffer_tokens >= pack_limit:
                for packed in pack_sequences(buffer, max_tokens_per_batch=pack_limit):
                    self._stats.batches_emitted += 1
                    yield packed
                buffer = []
                buffer_tokens = 0

    @property
    def stats(self) -> RLDatasetStats:
        return self._stats
