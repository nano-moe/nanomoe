# Data Pipeline

## Multi-stream packed data pipeline

```
HF IterableDataset
        │
        │ .shard(total_shards, index)
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
        │  pulls one PackedBatch from each stream,
        │  calls collate_packed_batches()
        ▼
   Single PackedBatch per step
   - tokens:        [total_tokens across all streams]
   - position_ids:  [per-doc positions, reset per doc]
   - cu_seqlens:    [0, ..., doc boundaries across all streams]
   - token_weights: [0 on first+last pos of each doc, 1 elsewhere]
   - stream_lengths: [tokens_in_stream_0, tokens_in_stream_1, ...]
        │
        ▼
   compute_loss() → model forward with document masking
```

## Key pieces

### PackedPretrainDataset

`src/nanomoe/data/packed_dataset.py`

One instance per stream. A background prefetch thread tokenizes examples from an
HF `IterableDataset`, packs documents greedily until `seq_len` tokens are
reached, and emits a `PackedBatch` through a `queue.Queue`. Labels are shifted
(`tokens[1:]`) and token weights are zero on the first and last position of each
document (first has no context, last has a filler label).

### PackedPretrainStreamGroup

`src/nanomoe/data/packed_dataset.py`

Shards the HF dataset across `world_size * batch_size` total shards. Each rank
gets `batch_size` streams starting at shard index `rank * batch_size`. Owns N
`PackedPretrainDataset` instances and yields collated batches via
`PackedBatchCollator`.

### collate_packed_batches()

`src/nanomoe/data/packing.py`

Concatenates N `PackedBatch` objects into one: cats tokens, position_ids, and
token_weights, then offsets `cu_seqlens` so document boundaries stay valid across
streams. Records `stream_lengths` and `stream_count` on the merged batch.
Validates invariants (tensor lengths, dtype/device consistency, optional field
presence).

### PackedBatchCollator

`src/nanomoe/data/packing.py`

Iterator that zips N streams, yields one collated batch per step, and guarantees
`stop()` on all streams via `try/finally`.

### Document masking

`cu_seqlens` drives two masking strategies:

- **`create_document_mask()`** — builds a dense 4D block-diagonal causal mask
  for standard HF attention.
- **`cu_seqlens_to_packing_metadata()`** — converts to `(doc_ids, seq_lens)` for
  `flex_attention` / `fsdp_attention` (block-sparse, no materialized mask).

Each document attends only within itself, causally.

### Sharding

`total_shards = world_size * batch_size`. Rank `r` gets shard indices
`[r * batch_size, ..., r * batch_size + batch_size - 1]`. No overlap, no
duplication across ranks.

## PackedBatch fields

| Field | Shape | Description |
|-------|-------|-------------|
| `tokens` | `[total_tokens]` | Flat concatenation of all documents across all streams |
| `labels` | `[total_tokens]` | Shifted tokens (`tokens[1:]` per doc, last is filler) |
| `position_ids` | `[total_tokens]` | Per-document positions, reset to 0 at each doc boundary |
| `cu_seqlens` | `[num_docs + 1]` | Cumulative sequence lengths marking document boundaries |
| `token_weights` | `[total_tokens]` | 1.0 for valid training positions, 0.0 for first/last per doc |
| `stream_lengths` | `list[int]` | Token count per stream (set by collation) |
| `stream_count` | `int` | Number of streams collated |
| `log_probs` | `[total_tokens]` or None | For RL pipelines |
| `rewards` | `[num_docs]` or None | For RL pipelines |
