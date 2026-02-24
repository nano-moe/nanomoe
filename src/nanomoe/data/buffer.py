"""Batch mixer that pulls from multiple sample sources and emits PackedBatch."""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from nanomoe.data.packing import pack_sequences
from nanomoe.data.types import PackedBatch, Sample


class SampleSource(Protocol):
    def __iter__(self) -> Iterator[Sample]: ...


class SourceSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    source: Any
    weight: float = 1.0

    @field_validator("source")
    @classmethod
    def _ensure_iterable(cls, value: Any) -> Any:
        try:
            iter(value)
        except TypeError as exc:
            raise TypeError("source must be iterable") from exc
        return value


class DataBufferConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    packed_seq_len: int = 4096  # Target tokens per packed sequence; max_seq_len controls per-document truncation
    max_tokens_per_batch: int | None = None
    prefetch_batches: int = 0
    seed: int = 42
    max_attempts: int = 10_000
    skip_zero_weight: bool = True


class DataBufferStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    samples_seen: int = 0
    skipped_zero_weight: int = 0
    batches_emitted: int = 0


@dataclass
class _SourceState:
    name: str
    weight: float
    source: Iterable[Sample]
    iterator: Iterator[Sample]
    exhausted: bool = False
    samples_seen: int = 0


class DataBuffer:
    """Mix multiple Sample sources and yield packed batches."""

    def __init__(self, sources: list[SourceSpec], config: DataBufferConfig | None = None):
        self.config = config or DataBufferConfig()
        self._rng = random.Random(self.config.seed)
        self._stats = DataBufferStats()

        self._sources: list[_SourceState] = []
        for spec in sources:
            self._sources.append(
                _SourceState(
                    name=spec.name,
                    weight=spec.weight,
                    source=spec.source,
                    iterator=iter(spec.source),
                )
            )

        self._queue: queue.Queue[PackedBatch | None] | None = None
        self._stop_event = threading.Event()
        self._prefetch_thread: threading.Thread | None = None

    def _sample_weight_sum(self, sample: Sample) -> float:
        if sample.token_weights:
            return float(sum(abs(w) for w in sample.token_weights))
        if sample.loss_mask:
            return float(sum(sample.loss_mask[1:]))
        return 0.0

    def _pick_source(self) -> _SourceState | None:
        active = [s for s in self._sources if not s.exhausted]
        if not active:
            return None
        total = sum(s.weight for s in active)
        r = self._rng.random() * total
        acc = 0.0
        for s in active:
            acc += s.weight
            if r <= acc:
                return s
        return active[-1]

    def _next_sample(self) -> Sample | None:
        while True:
            source = self._pick_source()
            if source is None:
                return None
            try:
                sample = next(source.iterator)
                source.samples_seen += 1
                self._stats.samples_seen += 1
                return sample
            except StopIteration:
                source.exhausted = True

    def _iter_batches(self) -> Iterator[PackedBatch]:
        buffer: list[Sample] = []
        buffer_tokens = 0
        attempts = 0
        pack_limit = self.config.max_tokens_per_batch or self.config.packed_seq_len

        while True:
            sample = self._next_sample()
            if sample is None:
                if buffer:
                    for packed in pack_sequences(buffer, max_tokens_per_batch=pack_limit):
                        self._stats.batches_emitted += 1
                        yield packed
                return

            attempts += 1
            if self.config.skip_zero_weight and self._sample_weight_sum(sample) <= 0:
                self._stats.skipped_zero_weight += 1
                if attempts >= self.config.max_attempts:
                    raise RuntimeError("Too many skipped samples (zero weight).")
                continue

            buffer.append(sample)
            buffer_tokens += len(sample.tokens)

            if buffer_tokens >= pack_limit:
                for packed in pack_sequences(buffer, max_tokens_per_batch=pack_limit):
                    self._stats.batches_emitted += 1
                    yield packed
                buffer = []
                buffer_tokens = 0
                attempts = 0

    def _prefetch_worker(self):
        try:
            for batch in self._iter_batches():
                if self._stop_event.is_set():
                    break
                assert self._queue is not None
                self._queue.put(batch)
        finally:
            if self._queue is not None:
                self._queue.put(None)

    def start(self):
        if self.config.prefetch_batches <= 0:
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.config.prefetch_batches)
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue = None

    def __iter__(self) -> Iterator[PackedBatch]:
        if self.config.prefetch_batches <= 0:
            yield from self._iter_batches()
            return

        self.start()
        assert self._queue is not None

        try:
            while True:
                batch = self._queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            self.stop()

    @property
    def stats(self) -> DataBufferStats:
        return self._stats

    def _reset_iterators(self) -> None:
        for source in self._sources:
            source.iterator = iter(source.source)
            source.exhausted = False
            for _ in range(source.samples_seen):
                try:
                    next(source.iterator)
                except StopIteration:
                    source.exhausted = True
                    break

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self._rng.getstate(),
            "stats": self._stats.model_dump(),
            "sources": {s.name: {"samples_seen": s.samples_seen, "exhausted": s.exhausted} for s in self._sources},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)
        stats = state.get("stats")
        if stats is not None:
            self._stats = DataBufferStats(**stats)
        source_state = state.get("sources", {})
        for source in self._sources:
            saved = source_state.get(source.name)
            if saved is None:
                continue
            source.samples_seen = int(saved.get("samples_seen", 0))
            source.exhausted = bool(saved.get("exhausted", False))
        self._reset_iterators()


def create_sft_tokenize_fn(
    tokenizer: Any,
    max_seq_len: int,
    input_key: str = "messages",
) -> Callable[[dict], Sample | None]:
    """Create a tokenize function for SFT data with chat format.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        max_seq_len: Maximum sequence length
        input_key: Key for messages in the dataset

    Returns:
        Tokenize function that creates Sample objects
    """

    def tokenize(example: dict) -> Sample | None:
        messages = example.get(input_key, [])
        if not messages:
            return None

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        # Find where the assistant response starts for loss masking
        # For now, simple heuristic: loss on everything after first assistant turn
        loss_mask = [0] * len(tokens)

        # Encode just the prompt (all messages except last assistant)
        prompt_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                break
            prompt_messages.append(msg)

        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
        else:
            prompt_len = 0

        # Set loss mask: 1 for response tokens
        for i in range(prompt_len, len(tokens)):
            loss_mask[i] = 1

        return Sample(
            tokens=tokens,
            loss_mask=loss_mask,
            token_weights=[float(x) for x in loss_mask[1:]] + [0.0],
            prompt_len=prompt_len,
        )

    return tokenize
