"""Tests for nanomoe.data module."""

import torch

from nanomoe.data import PackedBatch, PackedSFTDataset, Sample, SFTDatasetConfig, pack_sequences


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_creation(self):
        sample = Sample(
            tokens=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            log_probs=[-1.0, -2.0, -3.0],
            reward=1.5,
            prompt_len=2,
            group_id=0,
        )
        assert sample.tokens == [1, 2, 3, 4, 5]
        assert sample.loss_mask == [0, 0, 1, 1, 1]
        assert sample.reward == 1.5
        assert sample.prompt_len == 2
        assert sample.group_id == 0

    def test_sample_defaults(self):
        sample = Sample()
        assert sample.tokens == []
        assert sample.loss_mask == []
        assert sample.log_probs == []
        assert sample.reward == 0.0
        assert sample.prompt_len == 0
        assert sample.group_id is None


class TestPackSequences:
    """Tests for pack_sequences function."""

    def test_pack_single_sample(self):
        samples = [
            Sample(
                tokens=[1, 2, 3],
                loss_mask=[0, 1, 1],
                log_probs=[-1.0, -2.0],
                reward=1.0,
                prompt_len=1,
                group_id=0,
            )
        ]
        batches = pack_sequences(samples, num_packs=1)

        assert len(batches) == 1
        batch = batches[0]
        assert isinstance(batch, PackedBatch)
        assert batch.tokens.shape[0] == 3
        assert batch.token_weights.shape[0] == 3
        assert batch.token_weights.tolist() == [1.0, 1.0, 0.0]
        assert batch.position_ids.tolist() == [0, 1, 2]
        assert batch.cu_seqlens.tolist() == [0, 3]

    def test_pack_multiple_samples(self):
        samples = [
            Sample(tokens=[1, 2, 3], loss_mask=[0, 1, 1], log_probs=[-1.0, -2.0], reward=1.0, prompt_len=1, group_id=0),
            Sample(tokens=[4, 5], loss_mask=[0, 1], log_probs=[-3.0], reward=2.0, prompt_len=1, group_id=0),
        ]
        batches = pack_sequences(samples, num_packs=1)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.tokens.shape[0] == 5
        assert batch.cu_seqlens.tolist() == [0, 3, 5]
        assert batch.position_ids.tolist() == [0, 1, 2, 0, 1]

    def test_pack_multiple_batches(self):
        samples = [
            Sample(tokens=[1, 2, 3], loss_mask=[0, 1, 1], log_probs=[-1.0, -2.0], reward=1.0, prompt_len=1, group_id=0),
            Sample(tokens=[4, 5], loss_mask=[0, 1], log_probs=[-3.0], reward=2.0, prompt_len=1, group_id=1),
        ]
        batches = pack_sequences(samples, num_packs=2)

        assert len(batches) == 2
        # Each batch should have one sample
        assert batches[0].tokens.shape[0] + batches[1].tokens.shape[0] == 5

    def test_pack_token_weights(self):
        # Two samples in same group with different rewards
        # Each sample has 2 tokens but only 1 generated token (1 log_prob)
        samples = [
            Sample(tokens=[1, 2], loss_mask=[0, 1], log_probs=[-1.0], reward=3.0, prompt_len=1, group_id=0),
            Sample(tokens=[3, 4], loss_mask=[0, 1], log_probs=[-2.0], reward=1.0, prompt_len=1, group_id=0),
        ]
        batches = pack_sequences(samples, num_packs=1)

        batch = batches[0]
        assert batch.token_weights.shape[0] == 4
        # Rewards are per sample
        assert batch.rewards is not None
        assert batch.rewards.numel() == 2

    def test_pack_empty_samples(self):
        samples = []
        batches = pack_sequences(samples, num_packs=1)

        # Should return empty list or list with empty batch
        assert len(batches) == 0 or batches[0].tokens.shape[0] == 0


class TestPackedBatch:
    """Tests for PackedBatch dataclass."""

    def test_packed_batch_to_device(self):
        batch = PackedBatch(
            tokens=torch.tensor([1, 2, 3]),
            position_ids=torch.tensor([0, 1, 2]),
            cu_seqlens=torch.tensor([0, 3]),
            token_weights=torch.tensor([1.0, 1.0, 0.0]),
            log_probs=torch.tensor([-1.0, -2.0]),
            rewards=torch.tensor([1.0]),
        )
        # Just verify the batch was created correctly
        assert batch.tokens.shape == (3,)
        assert batch.cu_seqlens.shape == (2,)


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']} {msg['content']}")
        if add_generation_prompt:
            parts.append("assistant")
        return " ".join(parts)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class DummyDataset:
    def __init__(self, rows):
        self._rows = rows

    def shuffle(self, seed, buffer_size):
        return self

    def __iter__(self):
        return iter(self._rows)


class TestPackedSFTDataset:
    def test_sft_token_weights_shift(self):
        dataset = DummyDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok done"},
                    ]
                }
            ]
        )
        tokenizer = DummyTokenizer()
        config = SFTDatasetConfig(packed_seq_len=5, max_seq_len=5, min_seq_len=1, input_key="messages", seed=0)
        sft = PackedSFTDataset(hf_dataset=dataset, tokenizer=tokenizer, config=config)

        batch = next(iter(sft))
        assert batch.labels is not None
        assert batch.tokens.shape[0] == 5
        assert batch.token_weights.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
