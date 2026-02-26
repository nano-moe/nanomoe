"""Sequence packing utilities for efficient training with variable-length sequences.

Adapted from slime/backends/fsdp_utils/data_packing.py and verl/utils/seqlen_balancing.py
"""

import heapq
import math
from collections.abc import Iterable, Sequence
from typing import cast

import torch

from nanomoe.data.types import PackedBatch, Sample


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = False,
) -> list[list[int]]:
    """Partition sequences into k groups with balanced total lengths.

    Uses the Karmarkar-Karp differencing algorithm for load balancing.

    Args:
        seqlen_list: Length of each sequence
        k_partitions: Number of partitions to create
        equal_size: If True, each partition must have equal number of items

    Returns:
        List of k partitions, each containing indices into seqlen_list
    """
    if len(seqlen_list) < k_partitions:
        raise ValueError(f"num items ({len(seqlen_list)}) < k_partitions ({k_partitions})")

    class Partition:
        def __init__(self):
            self.total = 0
            self.indices: list[int] = []

        def add(self, idx: int, length: int):
            self.indices.append(idx)
            self.total += length

        def merge(self, other: "Partition"):
            self.indices.extend(other.indices)
            self.total += other.total

        def __lt__(self, other):
            return (self.total, len(self.indices)) < (other.total, len(other.indices))

    # Sort by length descending for greedy assignment
    sorted_items = sorted(enumerate(seqlen_list), key=lambda x: -x[1])

    # Initialize partitions
    partitions = [Partition() for _ in range(k_partitions)]
    heap = [(p.total, i, p) for i, p in enumerate(partitions)]
    heapq.heapify(heap)

    # Greedy assignment: always add to partition with smallest total
    for idx, length in sorted_items:
        _, i, p = heapq.heappop(heap)
        p.add(idx, length)
        heapq.heappush(heap, (p.total, i, p))

    return [sorted(p.indices) for _, _, p in sorted(heap, key=lambda x: x[1])]


def pack_sequences(
    samples: list[Sample],
    max_tokens_per_batch: int | None = None,
    num_packs: int | None = None,
) -> list[PackedBatch]:
    """Pack variable-length sequences into dense batches.

    Args:
        samples: List of samples to pack
        max_tokens_per_batch: Maximum tokens per packed batch
        num_packs: Explicit number of packs (overrides max_tokens_per_batch)

    Returns:
        List of PackedBatch objects ready for training
    """
    if not samples:
        return []

    seq_lengths = [len(s.tokens) for s in samples]

    # Determine number of packs
    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_batch:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_batch))
    else:
        k_partitions = 1

    # Get balanced partitions
    partitions = get_seqlen_balanced_partitions(seq_lengths, k_partitions)

    # Pack each partition
    result = []
    for indices in partitions:
        cu_seqlens = [0]
        flat_tokens = []
        flat_labels = []
        flat_token_weights = []
        flat_position_ids = []
        flat_log_probs = []
        rewards = []

        for i in indices:
            sample = samples[i]
            seq_len = len(sample.tokens)

            flat_tokens.extend(sample.tokens)
            if seq_len > 0:
                flat_labels.extend(sample.tokens[1:] + [sample.tokens[-1]])
            flat_position_ids.extend(range(seq_len))
            flat_log_probs.extend(sample.log_probs)
            rewards.append(sample.reward)

            if sample.token_weights:
                if len(sample.token_weights) != seq_len:
                    raise ValueError("token_weights length must match tokens length")
                flat_token_weights.extend(sample.token_weights)
            else:
                shifted = [float(x) for x in sample.loss_mask[1:]]
                shifted.append(0.0)
                flat_token_weights.extend(shifted)

            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        log_probs = torch.tensor(flat_log_probs, dtype=torch.float32) if flat_log_probs else None
        rewards_t = torch.tensor(rewards, dtype=torch.float32) if rewards else None

        packed = PackedBatch(
            tokens=torch.tensor(flat_tokens, dtype=torch.long),
            labels=torch.tensor(flat_labels, dtype=torch.long) if flat_labels else None,
            position_ids=torch.tensor(flat_position_ids, dtype=torch.long),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            token_weights=torch.tensor(flat_token_weights, dtype=torch.float32),
            log_probs=log_probs,
            rewards=rewards_t,
        )
        result.append(packed)

    return result


def unpack_batch(packed: PackedBatch) -> list[dict]:
    """Unpack a PackedBatch back into individual sequences.

    Useful for debugging or when per-sequence operations are needed.
    """
    cu_seqlens = packed.cu_seqlens.tolist()
    num_seqs = len(cu_seqlens) - 1

    sequences = []
    for i in range(num_seqs):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        sequences.append(
            {
                "tokens": packed.tokens[start:end],
                "labels": packed.labels[start:end] if packed.labels is not None else None,
                "token_weights": packed.token_weights[start:end],
                "position_ids": packed.position_ids[start:end],
                "reward": packed.rewards[i] if packed.rewards is not None else None,
            }
        )

    return sequences


def collate_packed_batches(batches: Sequence[PackedBatch]) -> PackedBatch:
    """Collate multiple PackedBatch objects from independent streams into one.

    Each stream packs independently to seq_len tokens with its own cu_seqlens,
    so we collate at the PackedBatch level (not token level). This keeps packing
    logic per-stream and makes collation a simple concatenation with cu_seqlens
    offset — streams must not pack across shard boundaries.

    Concatenates tokens, position_ids, token_weights, and offsets cu_seqlens
    so document boundaries remain valid across the merged batch.

    Args:
        batches: Sequence of PackedBatch from independent streams.
            All must be on the same device. Optional fields (labels, log_probs,
            rewards) must be all-present or all-None across batches.

    Returns:
        A single merged PackedBatch with stream_lengths and stream_count set.
    """
    if not batches:
        raise ValueError("Cannot collate empty sequence of batches")

    # Validate per-batch invariants
    ref_device = batches[0].tokens.device
    ref_dtype = batches[0].tokens.dtype
    for i, b in enumerate(batches):
        n = b.tokens.shape[0]
        if i > 0:
            if b.tokens.device != ref_device:
                raise ValueError(f"Device mismatch: batch 0 on {ref_device}, batch {i} on {b.tokens.device}")
            if b.tokens.dtype != ref_dtype:
                raise ValueError(f"Dtype mismatch: batch 0 is {ref_dtype}, batch {i} is {b.tokens.dtype}")
        if b.position_ids.shape[0] != n:
            raise ValueError(f"Batch {i}: position_ids length {b.position_ids.shape[0]} != tokens length {n}")
        if b.token_weights.shape[0] != n:
            raise ValueError(f"Batch {i}: token_weights length {b.token_weights.shape[0]} != tokens length {n}")
        if b.labels is not None and b.labels.shape[0] != n:
            raise ValueError(f"Batch {i}: labels length {b.labels.shape[0]} != tokens length {n}")
        cu_end = int(b.cu_seqlens[-1].item())
        if cu_end != n:
            raise ValueError(f"Batch {i}: cu_seqlens end {cu_end} != tokens length {n}")
        if int(b.cu_seqlens[0].item()) != 0:
            raise ValueError(f"Batch {i}: cu_seqlens must start at 0, got {int(b.cu_seqlens[0].item())}")
        diffs = b.cu_seqlens[1:] - b.cu_seqlens[:-1]
        if (diffs < 0).any():
            raise ValueError(f"Batch {i}: cu_seqlens is not monotonically non-decreasing")
        num_docs = b.cu_seqlens.shape[0] - 1
        if b.rewards is not None and b.rewards.shape[0] != num_docs:
            raise ValueError(f"Batch {i}: rewards length {b.rewards.shape[0]} != num_docs {num_docs}")

    # Validate optional fields: all-present or all-None
    for field_name in ("labels", "log_probs", "rewards"):
        present = [getattr(b, field_name) is not None for b in batches]
        if any(present) and not all(present):
            raise ValueError(
                f"Mixed presence of '{field_name}' across batches: got {sum(present)}/{len(batches)} present"
            )

    # Concatenate tensors
    tokens = torch.cat([b.tokens for b in batches])
    position_ids = torch.cat([b.position_ids for b in batches])
    token_weights = torch.cat([b.token_weights for b in batches])

    # Optional tensors
    labels = torch.cat(cast(list[torch.Tensor], [b.labels for b in batches])) if batches[0].labels is not None else None
    log_probs = (
        torch.cat(cast(list[torch.Tensor], [b.log_probs for b in batches]))
        if batches[0].log_probs is not None
        else None
    )
    rewards = (
        torch.cat(cast(list[torch.Tensor], [b.rewards for b in batches])) if batches[0].rewards is not None else None
    )

    # Offset cu_seqlens: first batch keeps full cu_seqlens, subsequent drop leading 0
    parts = [batches[0].cu_seqlens]
    offset = int(batches[0].cu_seqlens[-1].item())
    for b in batches[1:]:
        parts.append(b.cu_seqlens[1:] + offset)
        offset += int(b.cu_seqlens[-1].item())
    cu_seqlens = torch.cat(parts)

    stream_lengths = [int(b.cu_seqlens[-1].item()) for b in batches]

    return PackedBatch(
        tokens=tokens,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        token_weights=token_weights,
        labels=labels,
        log_probs=log_probs,
        rewards=rewards,
        stream_lengths=stream_lengths,
        stream_count=len(batches),
    )


class PackedBatchCollator:
    """Iterate over multiple PackedBatch streams, collating one batch from each per step."""

    def __init__(self, streams: Sequence[Iterable[PackedBatch]]):
        self.streams = streams

    @staticmethod
    def _close_streams(streams: Sequence) -> None:
        """Stop any streams that have a stop() method (e.g. PackedPretrainDataset)."""
        for s in streams:
            if hasattr(s, "stop"):
                s.stop()

    def __iter__(self):
        try:
            iterators = [iter(s) for s in self.streams]
            while True:
                try:
                    batches = [next(it) for it in iterators]
                except StopIteration:
                    return
                yield collate_packed_batches(batches)
        finally:
            self._close_streams(self.streams)
