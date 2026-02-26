# 2026-02-24: Multi-Stream Packed Data Loading

## Summary

Added `batch_size > 1` support for pretraining by running multiple independent
packed-sequence streams per rank, collated into a single `PackedBatch`. This
increases throughput without changing the model's forward signature.

## Architecture

```
HF IterableDataset
        │
        │ .shard(world_size * batch_size, rank * batch_size + i)
        ▼
┌─ PackedPretrainStreamGroup ──────────────────────┐
│                                                   │
│  Stream 0: PackedPretrainDataset                  │
│    [prefetch thread] → tokenize → pack → queue    │
│                                                   │
│  Stream 1: PackedPretrainDataset                  │
│    [prefetch thread] → tokenize → pack → queue    │
│                                                   │
│  ...N streams (N = batch_size)                    │
│                                                   │
└───────────────────────────────────────────────────┘
        │
        │  PackedBatchCollator
        │  draws one PackedBatch from each stream,
        │  calls collate_packed_batches()
        ▼
   Single PackedBatch per step
   - tokens:        [total_tokens]       (flat across all streams)
   - position_ids:  [total_tokens]       (reset per document)
   - cu_seqlens:    [num_docs + 1]       (boundaries across all streams)
   - token_weights: [total_tokens]       (0 on first+last per doc)
   - stream_lengths: list[int]           (tokens per stream)
        │
        ▼
   compute_loss()
   ├─ flex_attention path: cu_seqlens → packing_doc_ids + packing_seq_lens
   └─ fsdp_attention path: cu_seqlens → dense 4D mask (batch_size=1 only)
```

## Key design decisions

**Sharding**: `total_shards = world_size * batch_size`. Rank `r` gets shard
indices `[r*batch_size .. r*batch_size + batch_size - 1]`. No overlap across
ranks or streams.

**Flat batch dim**: The model always sees `batch_dim=1`. Multiple streams are
concatenated into a single flat token sequence. Document masking via
`packing_doc_ids` prevents cross-document and cross-stream attention.

**Defaults**: `batch_size=2`, `seq_len=8192`, `gradient_accumulation=1`,
`attention_type=flex_attention`. Each GPU sees 2 x 8192 = ~16K tokens per step.
`batch_size=4` OOMs on a single H100 80GB during backward.

**flex_attention required for batch_size > 1**: `fsdp_attention` builds a dense
O(seq_len^2) mask. With 4 streams x 8192 tokens = 32K total, that mask would be
~2 GB. Validation raises a hard error for `batch_size > 1 + fsdp_attention`.

**RMSNorm float upcast**: RMSNorm upcasts to float32 and casts back. Required
because `torch.compile` on the attention path can promote Q/K via RoPE to
float32 while V stays bfloat16, and `flex_attention` rejects mixed dtypes.

## Files changed

| File | Change |
|------|--------|
| `src/nanomoe/data/packed_dataset.py` | `PackedPretrainStreamGroup`, `cu_seqlens_to_packing_metadata()`, prefetch retry fix, last-position weight fix |
| `src/nanomoe/data/packing.py` | `collate_packed_batches()` with invariant validation, `PackedBatchCollator` with stream cleanup |
| `src/nanomoe/data/types.py` | `stream_lengths`, `stream_count` on `PackedBatch` |
| `src/nanomoe/experiments/pretrain.py` | Multi-stream data loading, `validate_pretrain_config()`, `log_stream_shapes`, CUBLAS probe |
| `src/nanomoe/model/model.py` | RMSNorm float upcast for flex_attention compatibility |
| `scripts/sft.py` | Use `unified_loss()` instead of manual shift+CE |
| `tests/test_collator.py` | 17 new tests: collation, invariants, stream cleanup, pack weights, dtype |

## Bug fixes included

1. **Prefetch document drop**: Queue-full caused silent document loss. Fixed with retry loop.
2. **Last-position bogus label**: Weight=1 on filler label in multi-doc packs. Fixed with `weights[-1] = 0.0`.
3. **`log_stream_shapes` no-op**: List metrics dropped by train loop. Changed to per-stream numeric keys.
4. **Prefetch error deadlock**: `queue.put(None)` with no timeout. Fixed with drain + timeout + fallback.
5. **cu_seqlens validation**: Added monotonicity and start-at-zero checks.
6. **position_ids dtype**: `pack_sequences` used int32. Changed to int64.

## How to train

Single GPU:
```bash
WANDB_MODE=disabled uv run python -m nanomoe.experiments.pretrain --batch_size=1 --attention_type=fsdp_attention
```

DDP (8x H100):
```bash
WANDB_MODE=disabled uv run torchrun --nproc_per_node=8 -m nanomoe.experiments.pretrain --distributed=True
```

Defaults: `batch_size=2`, `seq_len=8192`, `grad_accum=1`, `flex_attention`, 1000 steps.

Override anything via CLI flags (uses `chz`):
```bash
uv run torchrun --nproc_per_node=8 -m nanomoe.experiments.pretrain \
  --distributed=True --batch_size=2 --seq_len=4096 --max_steps=500
```

## Validated

- 116 unit tests passing
- DDP training: 8x H100, seq_len=8192, batch_size=2, flex_attention
