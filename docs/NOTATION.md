# Project notation and tensor layout

Code, equations, and inspection output use the symbols below. The canonical hidden-state layout is **(B, d_emb, n)**: batch first, one token per column.

| Symbol | Meaning | Default model example |
| --- | --- | --- |
| B | Number of example chunks in a batch | 16 |
| n | Token positions in the current input example | Up to 64 |
| V | Vocabulary size, including reserved tokens | From tokenizer.V |
| d_emb | Features in each token/model representation | 64 |
| H | Number of attention heads | 4 |
| d_k | Query/key features per head, d_emb / H | 16 |
| d_ff | Intermediate feature width in the two-layer MLP | 256 |
| L | Complete Transformer blocks | 2 |
| n_max | Maximum supported input context length | 64 |

L counts blocks of LayerNorm, attention, LayerNorm, and MLP with residual paths. It does not count those four operations separately. The model also has a final LayerNorm after the L blocks.

`ModelConfig` uses `V`, `n_max`, `d_emb`, `H`, `L`, and `d_ff` as field names. Its `d_k` property is derived; d_emb must be divisible by H, and d_k must be even for RoPE. `rope_base` defaults to 10000. If omitted, d_ff defaults to 4*d_emb. `TrainConfig` holds B, n, learning rate, steps, clipping, evaluation cadence, and device. Shared defaults can be overridden in train.py or with CLI flags. `TrainConfig.B` sets batch size. Actual n comes from the input and may be smaller than n_max. The tokenizer reports `tokenizer.V`; corpus inspection uses `show_stats(n=64)` and `python -m scripts.inspect_tokenizer --n 64`.

## Index the data by meaning

```text
token_ids[b, i]       one integer token ID
X[b, f, i]            one feature value for token i in example b
X[b]                 (d_emb, n): one example, with feature rows and token columns
X[b, :, i]           (d_emb,): one token; integer indexing removes its column axis
X[b, :, i:i+1]       (d_emb, 1): the same token with its column axis retained
X[:, f, :]           (B, n): one feature coordinate across all examples/positions
```

For token IDs `(B,n)`, rows identify examples and columns identify token positions. After lookup `(B,d_emb,n)`, each ID selects a column from E. V is the number of available lookup entries; n is the number selected per example. The tokenizer stores strings and IDs; learned embedding values belong to the model.

## Why keep B first

Both layouts can show a d_emb-by-n example matrix:

```text
Batch first: X has (B, d_emb, n); select example b with X[b].
Batch last:  Y has (d_emb, n, B); select example b with Y[:, :, b].
Y = X.permute(1, 2, 0) reorders axes without changing their meaning.
```

PyTorch matrix multiplication treats the last two axes as matrix axes and preceding axes as batch axes. With B first, each matrix is already d_emb-by-n, so `W @ X` projects every token column. With B last, the last matrix axes would be n-by-B; you would need an axis reorder or a differently specified contraction. Layout also affects memory strides; permuting a tensor does not guarantee a contiguous copy or a speed improvement.

This is a storage/indexing convention, not a different model. Reordering axes preserves example matrices when the corresponding indexing and operations are updated consistently.

## Parameters and intermediate tensors

| Object | Shape |
| --- | --- |
| Token embedding table E | (d_emb, V) |
| Optional learned position table P, alternative to the current RoPE model | (d_emb, n_max) |
| Selected token vectors | (B, d_emb, n) |
| Optional selected position vectors | (1, d_emb, n), broadcast over B |
| LayerNorm mean and variance | (B, 1, n) |
| LayerNorm gamma and beta | (d_emb, 1) each |
| Per-head query, key, value_down weights | (d_k, d_emb) each |
| Per-head value_up weight | (d_emb, d_k) |
| Q, K, values after splitting H heads | (B, H, d_k, n) |
| Attention scores/probabilities A | (B, H, n_keys, n_queries): key rows, query columns |
| Head outputs | (B, H, d_k, n) |
| Each value_up result | (B, d_emb, n) |
| Sum of all head updates | (B, d_emb, n) |
| Stored FFN W1 / W2 | (d_ff, d_emb) / (d_emb, d_ff) |
| FFN expanded activations | (B, d_ff, n) |
| Projection bias | (d_out, 1), broadcast over B and n |
| Output logits / next-token targets | (B, V, n) / (B, n) |
| Unembedding weight W_vocab | (V, d_emb), E.T by default |

Weights have shape `(d_out,d_in)`. The custom Linear layer uses `weight @ X + bias`, with each bias a column `(d_out,1)`. For a single token column x, this is the usual `y = W @ x + b`; for a sentence, the same W acts on every token column. The FFN is `W2 @ GELU(W1 @ X + b1) + b2`.

Attention uses `scores = K.transpose(-2,-1) @ Q / sqrt(d_k)`, `A = softmax(masked_scores, dim=-2)`, and `head_output = values @ A`. Each head applies its own `value_up`; the H embedding-space updates are summed. This is equivalent to concatenating head outputs and applying an output matrix whose column blocks are the per-head value_up weights, with the per-head biases summed. A causal mask has key rows j and query columns i: entry (j,i) is allowed when j <= i, making the allowed region upper triangular. Each attention column sums to one.

LayerNorm also reduces over `dim=-2`, but its axis represents **embedding features**, not keys. A vocabulary softmax on `(B,V,n)` reduces over V (`dim=-2`). Always name what the axis means; columnwise does not mean the same data are normalized in these three operations.

## Position choices and future extensions

The current architecture uses RoPE inside attention and no additive position vectors at the input. A learned-position class remains available as an alternative component. The original paper used both sine and cosine and reported nearly identical translation results with learned positions; neither finding guarantees quality for another corpus. Learned P has a fixed trained range. Sinusoidal vectors can be calculated beyond the training range, but this alone does not guarantee useful extrapolation. See [Section 3.5](https://arxiv.org/html/1706.03762v7#S3.SS5).

[RoPE](https://arxiv.org/abs/2104.09864) rotates adjacent query/key feature pairs by their logical token positions: `q_i_rotated = R(i) @ q_i`, and similarly for k. Values are not rotated. The implementation is `RotaryPositionalEmbedding` in `src/layers.py`. For pair r and position p, the rotation angle is `p / rope_base**(2*r/d_k)`. Compare position variants by training each from initialization under otherwise matching settings.

Each feature pair has its own frequency `omega[r] = rope_base**(-2*r/d_k)`. All heads currently share this ladder; independent learned Q/K matrices let them use it differently. Fast and slow rotations offer different positional scales, without guaranteeing specific semantic or grammatical roles.

`KVCache` in `src/cache.py` stores **already rotated keys** and unrotated values, each `(B,H,d_k,n_cached)`, separately for each attention layer. It does not retain old queries. `cache.length(layer_idx)` supplies the next logical RoPE position. With n_new new tokens, the score/mask matrix is rectangular `(n_cached+n_new,n_new)`, so a query sees the cached prefix and legal positions in its new chunk. Each layer must have a unique layer_idx. Use `attention.eval()` and `torch.no_grad()` for cached inference; ordinary training remains uncached. [Caching reference](https://huggingface.co/docs/transformers/main/en/cache_explanation).

```python
attention = CausalSelfAttention(d_emb=8, H=2, layer_idx=0).eval()
cache = KVCache(n_max=64)
with torch.no_grad():
    prefix_update = attention(x[..., :3], cache=cache)  # (B, 8, 3)
    next_update = attention(x[..., 3:4], cache=cache)   # (B, 8, 1), position 3
cache.clear()  # Before a different prompt/batch or after changing model weights.
```

The cache assumes fixed batch membership/order, equal sequence lengths, and contiguous positions from zero. It appends ordinary tensors and rejects context overflow; it does not implement eviction or variable-length padded batches. The full model and `generate()` now pass one cache through all L blocks. After prefill, supplying IDs `(B,1)` makes every block's norms and MLP process `(B,d_emb,1)`. Old tokens' later-layer information is already represented in those layers' cached keys/values; old residual streams and MLP activations are not rerun or stored in this cache. [PagedAttention](https://arxiv.org/abs/2309.06180) can later manage storage in blocks while preserving these logical positions; paging itself is not implemented.

`TinyTransformer(ids)` returns raw logits `(B,V,n)` for all next-token targets. `TinyTransformer(ids, cache=cache, last_token_only=True)` returns `(B,V,1)`, projecting only the last new column into vocabulary space. Softmax/sampling live in `generate.py`; top-k retains up to k vocabulary candidates, normalizes over them, and samples one ID per example. At n_max, `generate()` clears/rebuilds from the cropped window, retaining full output history.

`ModelConfig.tie_embeddings=True` makes `unembedding_weight = token_embedding.weight.T`: a view of the same parameter, not a copied table. Input lookup and output scoring both update E. Setting it False registers an independent `(V,d_emb)` matrix. Both options use a separate `(V,1)` output bias. Sharing saves V*d_emb parameters. [Weight tying reference](https://arxiv.org/abs/1608.05859).

## Other useful symbols

- **d_v:** value-vector width per head. This project sets it equal to d_k; it need not be an independent configuration field.
- **d_in, d_out:** generic input/output feature widths used by Linear. They become d_emb, d_k, d_ff, or V depending on the projection.
- **b, i, j, f, h, ell:** indices for batch example, query/token position, key position, feature, head, and Transformer block. They select entries; they are not dimension sizes.
- **Q, K, values:** query, key, and value tensors. V remains the scalar vocabulary size.
- **E, P, W_Q, W_K, W_V, W_O, W1, W2, W_vocab:** learned parameter tables/matrices. The V subscript in W_V names the value projection; it is not a dimension size.
- **gamma, beta, epsilon:** LayerNorm scale, offset, and numerical-stability constant. The epsilon argument is named `eps` in code.
- **learning_rate, temperature:** optimizer step-size control and generation sampling temperature. They are scalar hyperparameters, not tensor axes.
- **w:** a local-attention window size if a local variant is added. It differs from n (actual input length) and n_max (model cap); the current implementation uses full causal attention.
- **IGNORE_INDEX = -100:** right-padded target positions excluded from cross-entropy and valid-target metrics. Input padding uses ID zero; causal masking isolates it from earlier valid positions.
- **total_tokens:** corpus token occurrences reported by the tokenizer. This is distinct from the per-example length n and the vocabulary size V.
