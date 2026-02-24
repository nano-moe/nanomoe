"""Packed dataset for pretraining with document masking and prefetching.

Handles:
- Streaming from HuggingFace datasets
- On-the-fly tokenization
- Packing documents to target size with cu_seqlens for document masking
- Background prefetching for efficient GPU utilization
"""

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor

from nanomoe.data.packing import PackedBatchCollator
from nanomoe.data.types import PackedBatch


class PackedPretrainDataset:
    """Dataset that packs documents for efficient pretraining.

    Features:
    - Streams from HuggingFace IterableDataset
    - Tokenizes on-the-fly
    - Packs multiple documents to target token count
    - Background prefetching with configurable queue size
    - Document masking via cu_seqlens

    Usage:
        dataset = PackedPretrainDataset(
            hf_dataset=load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)["train"],
            tokenizer=tokenizer,
            seq_len=8192,
            prefetch_batches=4,
        )

        for batch in dataset:
            # batch.tokens: [seq_len] packed tokens
            # batch.cu_seqlens: document boundaries
            loss = train_step(model, batch)
    """

    def __init__(
        self,
        hf_dataset: Any,  # HuggingFace IterableDataset
        tokenizer: Any,
        seq_len: int = 8192,  # Total tokens per packed sequence (not per document — see max_seq_len)
        max_seq_len: int | None = None,  # Max length per document (None = seq_len)
        text_key: str = "text",
        min_doc_len: int = 64,  # Skip very short documents
        prefetch_batches: int = 4,
        shuffle_buffer: int | None = 10_000,
        seed: int = 42,
        add_special_tokens: bool = False,
        tokenize_fn: Callable[[dict, Any], list[int]] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len or seq_len
        self.text_key = text_key
        self.min_doc_len = min_doc_len
        self.prefetch_batches = prefetch_batches
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.add_special_tokens = add_special_tokens

        # Custom tokenization function or default
        self._tokenize_fn = tokenize_fn or self._default_tokenize

        # State
        self._epoch = 0
        self._samples_seen = 0

        # Prefetch queue and thread
        self._queue: queue.Queue[PackedBatch | None] = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._prefetch_thread: threading.Thread | None = None
        self._error: Exception | None = None

    def _default_tokenize(self, example: dict, tokenizer: Any) -> list[int]:
        """Default tokenization: encode text, optionally with special tokens."""
        text = example.get(self.text_key, "")
        if not text:
            return []
        return tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

    def _get_stream_iter(self) -> Iterator:
        """Get a fresh iterator over the dataset."""
        if self.shuffle_buffer is not None and self.shuffle_buffer > 0:
            ds = self.hf_dataset.shuffle(seed=self.seed + self._epoch, buffer_size=self.shuffle_buffer)
            return iter(ds)
        return iter(self.hf_dataset)

    def _pack_documents(self, doc_tokens: list[list[int]]) -> PackedBatch:
        """Pack a list of tokenized documents into a single batch."""
        all_input_ids = []
        all_labels = []
        all_position_ids = []
        all_token_weights = []
        cu_seqlens = [0]

        for tokens in doc_tokens:
            seq_len = len(tokens)
            all_input_ids.extend(tokens)
            # Labels are shifted: predict next token
            all_labels.extend(tokens[1:] + [tokens[-1]])  # Last label is ignored anyway
            all_position_ids.extend(range(seq_len))
            # Loss on all tokens except first (no previous context) and last
            # (label is a duplicate filler — no valid next token to predict)
            weights = [0.0] + [1.0] * (seq_len - 1)
            weights[-1] = 0.0
            all_token_weights.extend(weights)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        return PackedBatch(
            tokens=torch.tensor(all_input_ids, dtype=torch.long),
            labels=torch.tensor(all_labels, dtype=torch.long),
            position_ids=torch.tensor(all_position_ids, dtype=torch.long),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            token_weights=torch.tensor(all_token_weights, dtype=torch.float32),
        )

    def _prefetch_worker(self):
        """Background worker that fills the prefetch queue."""
        try:
            stream_iter = self._get_stream_iter()
            current_pack: list[list[int]] = []
            current_tokens = 0

            while not self._stop_event.is_set():
                # Get next document
                try:
                    example = next(stream_iter)
                except StopIteration:
                    self._epoch += 1
                    stream_iter = self._get_stream_iter()
                    continue

                # Tokenize
                tokens = self._tokenize_fn(example, self.tokenizer)
                if len(tokens) < self.min_doc_len:
                    continue

                # Truncate if needed
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[: self.max_seq_len]

                self._samples_seen += 1

                # Check if adding this doc would exceed seq_len
                if current_tokens + len(tokens) > self.seq_len and current_pack:
                    # Pack current documents and put into queue
                    batch = self._pack_documents(current_pack)
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(batch, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    else:
                        return

                    current_pack = []
                    current_tokens = 0

                # Add document to current pack
                current_pack.append(tokens)
                current_tokens += len(tokens)

        except Exception as e:
            self._error = e
            # Best-effort signal: try to drain one slot then enqueue sentinel.
            # If put still fails after timeout, _error is set and the main
            # thread will notice on its next queue.get() timeout cycle.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put(None, timeout=5.0)
            except queue.Full:
                pass

    def start(self):
        """Start the prefetch thread."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return

        self._stop_event.clear()
        self._error = None
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def stop(self):
        """Stop the prefetch thread."""
        if not hasattr(self, "_stop_event"):
            return
        self._stop_event.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __iter__(self) -> Iterator[PackedBatch]:
        """Iterate over packed batches."""
        self.start()

        try:
            while True:
                if self._error is not None:
                    raise RuntimeError(f"Prefetch worker error: {self._error}") from self._error

                try:
                    batch = self._queue.get(timeout=10.0)
                except queue.Empty:
                    if self._error is not None:
                        raise RuntimeError(f"Prefetch worker error: {self._error}") from self._error
                    continue

                if batch is None:
                    if self._error is not None:
                        raise RuntimeError(f"Prefetch worker error: {self._error}") from self._error
                    break

                yield batch
        finally:
            self.stop()

    def __del__(self):
        self.stop()

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    @property
    def epoch(self) -> int:
        return self._epoch

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "epoch": self._epoch,
            "samples_seen": self._samples_seen,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        self._epoch = state.get("epoch", 0)
        self._samples_seen = state.get("samples_seen", 0)


class PackedPretrainStreamGroup:
    """Multi-stream packed dataset that shards data across N independent streams.

    Each stream is a separate PackedPretrainDataset with its own shard of the
    HF dataset. Streams are collated into a single PackedBatch per iteration step
    via PackedBatchCollator.

    Args:
        hf_dataset: HuggingFace IterableDataset (before sharding).
        tokenizer: Tokenizer instance.
        num_streams: Number of parallel streams (= batch_size).
        total_shards: Total shards across all ranks and streams (world_size * batch_size).
        shard_base_index: First shard index for this rank (rank * batch_size).
        seq_len: Total tokens per packed sequence per stream.
        **kwargs: Passed through to PackedPretrainDataset (max_seq_len, text_key, etc.).
    """

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        num_streams: int,
        total_shards: int,
        shard_base_index: int,
        seq_len: int = 8192,
        max_examples: int | None = None,
        **kwargs: Any,
    ):
        if not hasattr(hf_dataset, "shard"):
            raise ValueError(
                "Dataset does not support .shard(); cannot create multi-stream group. "
                "Use a HuggingFace IterableDataset that supports sharding."
            )

        self.num_streams = num_streams
        self.streams: list[PackedPretrainDataset] = []
        for i in range(num_streams):
            shard = hf_dataset.shard(num_shards=total_shards, index=shard_base_index + i)
            if max_examples is not None:
                shard = shard.take(max_examples)
            stream = PackedPretrainDataset(
                hf_dataset=shard,
                tokenizer=tokenizer,
                seq_len=seq_len,
                **kwargs,
            )
            self.streams.append(stream)

    def __iter__(self) -> Iterator[PackedBatch]:
        """Stop existing streams, restart, and yield collated batches."""
        # Stop any existing prefetch threads
        self.stop()
        # Yield from a fresh collator; ensure streams are stopped on exit/error
        try:
            yield from PackedBatchCollator(self.streams)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all stream prefetch threads."""
        for stream in self.streams:
            stream.stop()


def cu_seqlens_to_packing_metadata(cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
    """Convert cu_seqlens to packing_doc_ids and packing_seq_lens for flex_attention.

    Args:
        cu_seqlens: [num_docs + 1] cumulative sequence lengths

    Returns:
        packing_doc_ids:  [1, total_tokens] — unique doc ID per token
        packing_seq_lens: [1] — total token count (no padding)
    """
    device = cu_seqlens.device
    seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long, device=device)
    num_docs = int(seg_lens.shape[0])
    doc_ids = torch.repeat_interleave(torch.arange(num_docs, device=device), seg_lens).unsqueeze(0)  # [1, total_tokens]
    packing_seq_lens = cu_seqlens[-1:].to(dtype=torch.long)
    return doc_ids, packing_seq_lens


def create_document_mask(cu_seqlens: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Create a block-diagonal causal attention mask from cu_seqlens.

    Args:
        cu_seqlens: [num_docs + 1] cumulative sequence lengths
        dtype: Output dtype

    Returns:
        4D attention mask [1, 1, seq_len, seq_len] for HuggingFace models
        Values are 0 for allowed positions, -inf for masked positions
    """
    seq_len = int(cu_seqlens[-1].item())
    device = cu_seqlens.device

    # Create document IDs for each position
    doc_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        doc_ids[start:end] = i

    # Causal mask: position i can attend to positions j <= i
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    # Document mask: can only attend within same document
    same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)

    # Combined mask
    mask = causal_mask & same_doc

    # Convert to attention mask format (0 = attend, -inf = mask)
    mask_4d = mask.unsqueeze(0).unsqueeze(0).to(dtype)
    mask_4d = (1.0 - mask_4d) * torch.finfo(dtype).min

    return mask_4d
