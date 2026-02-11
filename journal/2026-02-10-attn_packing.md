In `tests/test_attention_packing.py`, we tried a bunch of ways to implement attention with
packing,

```
tests/test_attention_packing.py ..packed-attn benchmark (torch.bfloat16, tokens=2048, heads=16, head_dim=64)
cu_seqlens varlen SDPA: 0.447 ms/iter
flex attention       : 2.389 ms/iter
flex attention+compile: 0.056 ms/iter
4d attention mask    : 0.226 ms/iter
```

it seems that flex attention + compile is the fastest and I guess the reason could be it's full 
utlization of the sparsity in the attention mask.

Results on larger scale experiments
```
tests/test_attention_packing.py ..packed-attn benchmark (torch.bfloat16, tokens=7168, heads=16, head_dim=64)
cu_seqlens varlen SDPA: 1.321 ms/iter, peak=717.2 MB
flex attention       : 112.805 ms/iter, peak=41512.6 MB
flex attention+compile: 0.939 ms/iter, peak=687.7 MB
4d attention mask    : 10.639 ms/iter, peak=684.2 MB
4d mask+compile      : 10.547 ms/iter, peak=684.2 MB
```
show that flash attention (first row) is just good. We should stick with it.

The fallback option should be 4d attention mask + `F.scaled_dot_product_attention`.

-----

Ok it turns out that `cu_seqlens varlen SDPA` is for-looped based...so we should use flex attention

---

## Attention module design notes (packed sequences)

Recent benchmark (attention module, packed docs):

```
tests/test_attention_module.py attention module packed benchmark (torch.bfloat16, tokens=20000, hidden=1024)
fsdp(4d mask): 21.327 ms/iter, peak=2979.6 MB
flex(doc mask): 3.931 ms/iter, peak=2215.2 MB
fsdp/flex ratio: 5.425x
fsdp/flex peak memory ratio: 1.345x
```

### Proposed design

Goal: a single attention module that supports (1) packed sequences, (2) optional flex attention
for doc-level masking, (3) SDPA fallback for portability, and (4) simple integration with FSDP.

Key ideas:
- Prefer `flex_attention` + block mask when available (much faster and lower peak memory).
- Fallback to SDPA with a dense 4D additive mask for correctness.
- Avoid per-document loops in the forward path.
- Keep masking logic consistent across modes (doc-level causal).

### What should be in the arguments

**Core inputs**
- `q`, `k`, `v` tensors shaped `[B, H, T, D]`.
- `is_causal: bool` (default `True` for LM).

**Packed-sequence inputs**
- `cu_seqlens: Tensor | None` (int32 cumulative lengths for packed docs). If set, we build doc-aware masks.
- `total_tokens: int | None` (optional; derived from `cu_seqlens` when not provided).
- `doc_ids: Tensor | None` (optional alternative to `cu_seqlens` for precomputed doc mapping).

**Mask selection / backend**
- `attn_backend: Literal["flex", "sdpa"] | None` (auto-select when `None`).
- `block_mask: Tensor | None` (optional precomputed block mask to avoid recompute).
- `attn_mask_4d: Tensor | None` (optional precomputed 4D additive mask for SDPA fallback).

**Performance / compile options**
- `use_compile: bool` (default `False`), wrap flex path with `torch.compile` if available.
- `dropout_p: float` (training support).

**Return**
- Standard attention output tensor `[B, H, T, D]`.

### Behavior summary

If `cu_seqlens` (or `doc_ids`) is provided:
- Build doc-aware causal mask.
- Prefer flex attention when available; otherwise use SDPA + 4D mask.

If packed inputs are not provided:
- Use standard SDPA causal attention (or flex if desired).

This matches the benchmark results above: flex attention with doc masks is the fastest option
for packed sequences; SDPA + 4D mask is the portability fallback.