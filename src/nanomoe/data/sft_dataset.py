"""Packed dataset for supervised fine-tuning (SFT)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from nanomoe.data.buffer import create_sft_tokenize_fn
from nanomoe.data.types import PackedBatch, Sample


class SFTDatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    packed_seq_len: int = 2048  # Target tokens per packed sequence; max_seq_len controls per-document truncation
    max_seq_len: int | None = None
    min_seq_len: int = 16
    input_key: str = "messages"
    shuffle_buffer: int = 10_000
    seed: int = 42


class PackedSFTDataset:
    """Pack SFT samples into PackedBatch objects."""

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        config: SFTDatasetConfig | None = None,
        tokenize_fn: Callable[[dict], Sample | None] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.config = config or SFTDatasetConfig()

        max_len = self.config.max_seq_len or self.config.packed_seq_len
        self._tokenize_fn = tokenize_fn or create_sft_tokenize_fn(tokenizer, max_len, self.config.input_key)

        self._epoch = 0
        self._samples_seen = 0

    def _get_stream_iter(self) -> Iterator:
        ds = self.hf_dataset.shuffle(seed=self.config.seed + self._epoch, buffer_size=self.config.shuffle_buffer)
        return iter(ds)

    def _shift_loss_mask(self, loss_mask: list[int]) -> list[float]:
        if not loss_mask:
            return []
        shifted = [float(x) for x in loss_mask[1:]]
        shifted.append(0.0)
        return shifted

    def _pack_samples(self, samples: list[Sample]) -> PackedBatch:
        flat_tokens: list[int] = []
        flat_labels: list[int] = []
        flat_position_ids: list[int] = []
        flat_token_weights: list[float] = []
        cu_seqlens = [0]

        for sample in samples:
            tokens = sample.tokens
            if not tokens:
                continue
            seq_len = len(tokens)

            flat_tokens.extend(tokens)
            flat_labels.extend(tokens[1:] + [tokens[-1]])
            flat_position_ids.extend(range(seq_len))

            if sample.token_weights and len(sample.token_weights) == seq_len:
                flat_token_weights.extend(sample.token_weights)
            else:
                flat_token_weights.extend(self._shift_loss_mask(sample.loss_mask))

            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        return PackedBatch(
            tokens=torch.tensor(flat_tokens, dtype=torch.long),
            labels=torch.tensor(flat_labels, dtype=torch.long),
            position_ids=torch.tensor(flat_position_ids, dtype=torch.long),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            token_weights=torch.tensor(flat_token_weights, dtype=torch.float32),
        )

    def __iter__(self) -> Iterator[PackedBatch]:
        stream_iter = self._get_stream_iter()
        current_pack: list[Sample] = []
        current_tokens = 0

        while True:
            try:
                raw = next(stream_iter)
            except StopIteration:
                self._epoch += 1
                stream_iter = self._get_stream_iter()
                continue

            sample = self._tokenize_fn(raw)
            if sample is None or len(sample.tokens) < self.config.min_seq_len:
                continue

            self._samples_seen += 1

            if current_tokens + len(sample.tokens) > self.config.packed_seq_len and current_pack:
                yield self._pack_samples(current_pack)
                current_pack = []
                current_tokens = 0

            current_pack.append(sample)
            current_tokens += len(sample.tokens)

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    @property
    def epoch(self) -> int:
        return self._epoch
