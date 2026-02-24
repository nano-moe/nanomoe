"""Tests for RLDataset."""

import torch

from nanomoe.data import RLDataset, RLDatasetConfig, ScoredGroup


class DummySampler:
    def sample_and_score(self, prompt: torch.Tensor) -> ScoredGroup:
        _ = prompt
        tokens = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
        log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
        rewards = torch.tensor([1.0, 2.0])
        return ScoredGroup(tokens=tokens, log_probs=log_probs, rewards=rewards, prompt_len=2)


def test_rl_dataset_packs_group():
    prompts = [torch.tensor([1, 2])]
    cfg = RLDatasetConfig(seq_len=8, max_tokens_per_batch=8)
    dataset = RLDataset(prompts, DummySampler(), cfg)

    batch = next(iter(dataset))
    assert batch.labels is not None
    assert batch.tokens.shape[0] == 8
    assert batch.token_weights.shape[0] == 8
    assert batch.token_weights[:-1].abs().sum().item() > 0
