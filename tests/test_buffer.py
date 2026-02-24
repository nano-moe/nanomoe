"""Tests for DataBuffer mixing logic."""

from nanomoe.data.buffer import DataBuffer, DataBufferConfig, SourceSpec
from nanomoe.data.types import Sample


def _make_sample(tokens: list[int], weight: float = 1.0) -> Sample:
    loss_mask = [0] + [1] * (len(tokens) - 1)
    token_weights = [0.0] + [weight] * (len(tokens) - 1)
    return Sample(tokens=tokens, loss_mask=loss_mask, token_weights=token_weights)


def test_databuffer_mixes_sources():
    sources = [
        SourceSpec(name="a", source=[_make_sample([1, 2, 3])], weight=1.0),
        SourceSpec(name="b", source=[_make_sample([4, 5, 6])], weight=1.0),
    ]
    config = DataBufferConfig(max_tokens_per_batch=3, seq_len=3, prefetch_batches=0, seed=0)
    buf = DataBuffer(sources, config)

    batches = list(buf)
    assert len(batches) == 2
    assert all(b.tokens.shape[0] == 3 for b in batches)
    assert buf.stats.samples_seen == 2
    assert buf.stats.batches_emitted == 2


def test_databuffer_skips_zero_weight_samples():
    zero = _make_sample([1, 2, 3], weight=0.0)
    nonzero = _make_sample([4, 5, 6], weight=1.0)
    sources = [SourceSpec(name="a", source=[zero, nonzero], weight=1.0)]
    config = DataBufferConfig(max_tokens_per_batch=3, seq_len=3, prefetch_batches=0, seed=0)
    buf = DataBuffer(sources, config)

    batches = list(buf)
    assert len(batches) == 1
    assert buf.stats.skipped_zero_weight == 1


def test_databuffer_resume_replays_stream():
    sources = [
        SourceSpec(name="a", source=[_make_sample([1, 2, 3]), _make_sample([7, 8, 9])], weight=1.0),
        SourceSpec(name="b", source=[_make_sample([4, 5, 6]), _make_sample([10, 11, 12])], weight=1.0),
    ]
    config = DataBufferConfig(max_tokens_per_batch=3, seq_len=3, prefetch_batches=0, seed=123)
    buf = DataBuffer(sources, config)

    it = iter(buf)
    first = next(it)
    state = buf.state_dict()
    second = next(it)

    buf_replay = DataBuffer(sources, config)
    buf_replay.load_state_dict(state)
    second_replay = next(iter(buf_replay))

    assert first.tokens.shape == (3,)
    assert second.tokens.tolist() == second_replay.tokens.tolist()
