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