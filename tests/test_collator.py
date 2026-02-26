"""Tests for multi-stream collation and cu_seqlens conversion."""

import torch

from nanomoe.data.packed_dataset import (
    PackedPretrainDataset,
    cu_seqlens_to_packing_metadata,
)
from nanomoe.data.packing import PackedBatchCollator, collate_packed_batches, pack_sequences
from nanomoe.data.types import PackedBatch, Sample


def _make_batch(tokens, cu_seqlens, labels=None):
    n = len(tokens)
    pos = []
    seqlens = cu_seqlens
    for i in range(len(seqlens) - 1):
        seg_len = seqlens[i + 1] - seqlens[i]
        pos.extend(range(seg_len))
    return PackedBatch(
        tokens=torch.tensor(tokens, dtype=torch.long),
        position_ids=torch.tensor(pos, dtype=torch.long),
        cu_seqlens=torch.tensor(seqlens, dtype=torch.int32),
        token_weights=torch.ones(n, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long) if labels is not None else None,
    )


class TestCollatePackedBatches:
    def test_single_batch_passthrough(self):
        b = _make_batch([1, 2, 3, 4, 5], [0, 3, 5], labels=[2, 3, 0, 5, 0])
        merged = collate_packed_batches([b])
        assert merged.tokens.tolist() == [1, 2, 3, 4, 5]
        assert merged.cu_seqlens.tolist() == [0, 3, 5]
        assert merged.stream_lengths == [5]
        assert merged.stream_count == 1

    def test_two_batches_offsets_cu_seqlens(self):
        b1 = _make_batch([1, 2, 3], [0, 2, 3])  # 2 docs: len 2 + len 1
        b2 = _make_batch([4, 5, 6, 7], [0, 4])  # 1 doc: len 4
        merged = collate_packed_batches([b1, b2])

        assert merged.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
        # b1 cu_seqlens = [0, 2, 3], b2 cu_seqlens = [0, 4]
        # merged = [0, 2, 3] + [3+4] = [0, 2, 3, 7]
        assert merged.cu_seqlens.tolist() == [0, 2, 3, 7]
        assert merged.stream_lengths == [3, 4]
        assert merged.stream_count == 2
        assert merged.position_ids.tolist() == [0, 1, 0, 0, 1, 2, 3]

    def test_three_batches(self):
        b1 = _make_batch([10, 20], [0, 2])
        b2 = _make_batch([30, 40, 50], [0, 1, 3])
        b3 = _make_batch([60], [0, 1])
        merged = collate_packed_batches([b1, b2, b3])

        assert merged.tokens.tolist() == [10, 20, 30, 40, 50, 60]
        # [0, 2] + [2+1, 2+3] + [5+1] = [0, 2, 3, 5, 6]
        assert merged.cu_seqlens.tolist() == [0, 2, 3, 5, 6]
        assert merged.stream_lengths == [2, 3, 1]
        assert merged.stream_count == 3

    def test_labels_all_present(self):
        b1 = _make_batch([1, 2], [0, 2], labels=[2, 0])
        b2 = _make_batch([3, 4], [0, 2], labels=[4, 0])
        merged = collate_packed_batches([b1, b2])
        assert merged.labels is not None
        assert merged.labels.tolist() == [2, 0, 4, 0]

    def test_labels_all_none(self):
        b1 = _make_batch([1, 2], [0, 2])
        b2 = _make_batch([3, 4], [0, 2])
        merged = collate_packed_batches([b1, b2])
        assert merged.labels is None

    def test_labels_mixed_raises(self):
        b1 = _make_batch([1, 2], [0, 2], labels=[2, 0])
        b2 = _make_batch([3, 4], [0, 2])  # no labels
        try:
            collate_packed_batches([b1, b2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "labels" in str(e)

    def test_empty_raises(self):
        try:
            collate_packed_batches([])
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_stream_fields_survive_to(self):
        b1 = _make_batch([1, 2], [0, 2])
        b2 = _make_batch([3, 4], [0, 2])
        merged = collate_packed_batches([b1, b2])
        moved = merged.to(torch.device("cpu"))
        assert moved.stream_lengths == [2, 2]
        assert moved.stream_count == 2


class TestPackedBatchCollator:
    def test_yields_collated_batches(self):
        b1a = _make_batch([1, 2], [0, 2])
        b1b = _make_batch([3, 4], [0, 2])
        b2a = _make_batch([5, 6, 7], [0, 3])
        b2b = _make_batch([8, 9, 10], [0, 3])

        stream1 = [b1a, b1b]
        stream2 = [b2a, b2b]
        collator = PackedBatchCollator([stream1, stream2])

        batches = list(collator)
        assert len(batches) == 2
        assert batches[0].tokens.tolist() == [1, 2, 5, 6, 7]
        assert batches[1].tokens.tolist() == [3, 4, 8, 9, 10]
        assert batches[0].stream_count == 2


class TestCuSeqlensToPackingMetadata:
    def test_single_doc(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        doc_ids, seq_lens = cu_seqlens_to_packing_metadata(cu)
        assert doc_ids.shape == (1, 5)
        assert doc_ids.tolist() == [[0, 0, 0, 0, 0]]
        assert seq_lens.tolist() == [5]

    def test_multiple_docs(self):
        cu = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        doc_ids, seq_lens = cu_seqlens_to_packing_metadata(cu)
        assert doc_ids.shape == (1, 8)
        assert doc_ids.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2]]
        assert seq_lens.tolist() == [8]

    def test_device_inferred(self):
        cu = torch.tensor([0, 2, 4], dtype=torch.int32)
        doc_ids, seq_lens = cu_seqlens_to_packing_metadata(cu)
        assert doc_ids.device == cu.device
        assert seq_lens.device == cu.device

    def test_output_shapes(self):
        cu = torch.tensor([0, 10, 20, 25], dtype=torch.int32)
        doc_ids, seq_lens = cu_seqlens_to_packing_metadata(cu)
        assert doc_ids.dim() == 2
        assert doc_ids.shape[0] == 1
        assert doc_ids.shape[1] == 25
        assert seq_lens.shape == (1,)


class TestPretrainValidation:
    def test_batch_size_gt1_fsdp_raises(self):
        """batch_size > 1 with fsdp_attention must raise ValueError."""
        from nanomoe.experiments.pretrain import TrainConfig, validate_pretrain_config
        from nanomoe.model import MoEConfig

        cfg = TrainConfig(batch_size=4, attention_type="fsdp_attention")
        model_config = MoEConfig.tiny()
        model_config.attention_type = "fsdp_attention"

        try:
            validate_pretrain_config(cfg, model_config)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "fsdp_attention" in str(e)

    def test_batch_size_1_fsdp_ok(self):
        """batch_size == 1 with fsdp_attention should be fine."""
        from nanomoe.experiments.pretrain import TrainConfig, validate_pretrain_config
        from nanomoe.model import MoEConfig

        cfg = TrainConfig(batch_size=1, attention_type="fsdp_attention")
        model_config = MoEConfig.tiny()
        model_config.attention_type = "fsdp_attention"
        validate_pretrain_config(cfg, model_config)  # should not raise

    def test_batch_size_zero_raises(self):
        """batch_size < 1 must raise ValueError."""
        from nanomoe.experiments.pretrain import TrainConfig, validate_pretrain_config
        from nanomoe.model import MoEConfig

        cfg = TrainConfig(batch_size=0)
        model_config = MoEConfig.tiny()

        try:
            validate_pretrain_config(cfg, model_config)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "batch_size" in str(e)


class TestCollateInvariantValidation:
    def test_mixed_dtype_raises(self):
        """Collating batches with different token dtypes must raise."""
        b1 = PackedBatch(
            tokens=torch.tensor([1, 2], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
        )
        b2 = PackedBatch(
            tokens=torch.tensor([3, 4], dtype=torch.int32),  # different dtype
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
        )
        try:
            collate_packed_batches([b1, b2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Dtype" in str(e)

    def test_position_ids_length_mismatch_raises(self):
        """position_ids must match tokens length."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),  # too short
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            token_weights=torch.ones(3),
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "position_ids" in str(e)

    def test_token_weights_length_mismatch_raises(self):
        """token_weights must match tokens length."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            token_weights=torch.ones(2),  # too short
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "token_weights" in str(e)

    def test_cu_seqlens_end_mismatch_raises(self):
        """cu_seqlens[-1] must equal tokens length."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),  # ends at 2, not 3
            token_weights=torch.ones(3),
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "cu_seqlens" in str(e)

    def test_labels_length_mismatch_raises(self):
        """labels must match tokens length when present."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            token_weights=torch.ones(3),
            labels=torch.tensor([2, 3], dtype=torch.long),  # too short
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "labels" in str(e)

    def test_rewards_length_mismatch_raises(self):
        """rewards must match num_docs (cu_seqlens.shape[0] - 1)."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),  # 2 docs
            token_weights=torch.ones(5),
            rewards=torch.tensor([1.0, 2.0, 3.0]),  # 3 rewards but only 2 docs
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "rewards" in str(e)

    def test_cu_seqlens_non_monotonic_raises(self):
        """cu_seqlens must be monotonically non-decreasing."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 3, 2, 5], dtype=torch.int32),  # 3 > 2
            token_weights=torch.ones(5),
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "monoton" in str(e).lower()

    def test_cu_seqlens_not_starting_at_zero_raises(self):
        """cu_seqlens must start at 0."""
        b = PackedBatch(
            tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            cu_seqlens=torch.tensor([1, 3], dtype=torch.int32),  # starts at 1
            token_weights=torch.ones(3),
        )
        try:
            collate_packed_batches([b])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "start at 0" in str(e)

    def test_log_probs_mixed_raises(self):
        """Mixed presence of log_probs across batches must raise."""
        b1 = PackedBatch(
            tokens=torch.tensor([1, 2], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
            log_probs=torch.tensor([0.1, 0.2]),
        )
        b2 = PackedBatch(
            tokens=torch.tensor([3, 4], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
            # no log_probs
        )
        try:
            collate_packed_batches([b1, b2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "log_probs" in str(e)

    def test_rewards_mixed_raises(self):
        """Mixed presence of rewards across batches must raise."""
        b1 = PackedBatch(
            tokens=torch.tensor([1, 2], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
            rewards=torch.tensor([1.0]),
        )
        b2 = PackedBatch(
            tokens=torch.tensor([3, 4], dtype=torch.long),
            position_ids=torch.tensor([0, 1], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            token_weights=torch.ones(2),
            # no rewards
        )
        try:
            collate_packed_batches([b1, b2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "rewards" in str(e)


class TestCollatorStreamCleanup:
    def test_collator_calls_stop_on_streams(self):
        """PackedBatchCollator should call stop() on streams when iteration ends."""
        stop_called = []

        class MockStream:
            def __init__(self, batches):
                self._batches = batches

            def __iter__(self):
                return iter(self._batches)

            def stop(self):
                stop_called.append(True)

        b1 = _make_batch([1, 2], [0, 2])
        b2 = _make_batch([3, 4], [0, 2])
        s1 = MockStream([b1])
        s2 = MockStream([b2])

        collator = PackedBatchCollator([s1, s2])
        list(collator)  # exhaust
        assert len(stop_called) == 2

    def test_collator_calls_stop_on_error(self):
        """PackedBatchCollator should call stop() even if iteration raises."""
        stop_called = []

        class ErrorStream:
            def __iter__(self):
                raise RuntimeError("boom")

            def stop(self):
                stop_called.append(True)

        class GoodStream:
            def __iter__(self):
                return iter([_make_batch([1, 2], [0, 2])])

            def stop(self):
                stop_called.append(True)

        collator = PackedBatchCollator([GoodStream(), ErrorStream()])
        try:
            list(collator)
        except RuntimeError:
            pass
        assert len(stop_called) == 2


class TestPretrainPackWeights:
    """Verify that _pack_documents sets weight=0 on the last position of each document."""

    def test_single_doc_last_position_zero_weight(self):
        """Single doc: first and last positions should have weight 0."""
        ds = PackedPretrainDataset.__new__(PackedPretrainDataset)
        batch = ds._pack_documents([[10, 20, 30, 40]])
        # weights: [0, 1, 1, 0]
        assert batch.token_weights.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_multi_doc_last_positions_zero_weight(self):
        """Multiple docs: last position of EACH doc should have weight 0."""
        ds = PackedPretrainDataset.__new__(PackedPretrainDataset)
        batch = ds._pack_documents([[10, 20, 30], [40, 50]])
        # doc1 weights: [0, 1, 0], doc2 weights: [0, 0]
        assert batch.token_weights.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]

    def test_two_token_doc_has_all_zero_weights(self):
        """A 2-token doc has no valid training positions (first=no context, last=no next token)."""
        ds = PackedPretrainDataset.__new__(PackedPretrainDataset)
        batch = ds._pack_documents([[10, 20]])
        assert batch.token_weights.tolist() == [0.0, 0.0]

    def test_one_token_doc_has_zero_weight(self):
        """A 1-token doc has weight 0 (it's both first and last)."""
        ds = PackedPretrainDataset.__new__(PackedPretrainDataset)
        batch = ds._pack_documents([[10]])
        assert batch.token_weights.tolist() == [0.0]


class TestPackSequencesPositionIdsDtype:
    """Verify position_ids use int64 (torch.long) consistently."""

    def test_position_ids_dtype_is_long(self):
        sample = Sample(
            tokens=[1, 2, 3, 4],
            loss_mask=[0, 0, 1, 1],
        )
        batches = pack_sequences([sample], num_packs=1)
        assert len(batches) == 1
        assert batches[0].position_ids.dtype == torch.long
