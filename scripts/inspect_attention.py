"""Print corpus tokens, RoPE, attention probabilities, and incremental KV caching.

Run: .venv/bin/python -m scripts.inspect_attention
"""

import json
import math
from pathlib import Path

import torch

from scripts.inspect_layers import print_token_columns
from src.attention import CausalSelfAttention, causal_mask
from src.cache import KVCache
from src.layers import TokenEmbedding
from src.tokenizer import Tokenizer


def main() -> None:
    torch.manual_seed(42)
    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "gutenberg"
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = sorted(corpus_dir / item["path"] for item in manifest["documents"]
                   if item["split"] == "train")
    tokenizer = Tokenizer.from_files(paths)
    first_book = paths[0].stem.split("_")[0]
    other = next(path for path in paths if path.stem.split("_")[0] != first_book)
    texts = [path.read_text(encoding="utf-8").split("\n\n", 1)[1].lstrip()
             for path in (paths[0], other)]  # B=2 text samples.
    B, n, d_emb, H = 2, 6, 8, 2  # Scalars; d_k = 4.
    ids = torch.tensor([tokenizer.encode(text)[:n] for text in texts])  # (B, n).
    tokens = tokenizer.tokenize(texts[0])[:n]  # n strings labeling example zero.
    embedding = TokenEmbedding(tokenizer.V, d_emb)
    attention = CausalSelfAttention(d_emb, H, layer_idx=0).eval()
    cache = KVCache(n_max=64)

    with torch.no_grad():
        x = embedding(ids)  # (B, d_emb, n); no additive positional embeddings.
        output, probabilities = attention(x, return_attention=True)
        # output: (B, d_emb, n); probabilities: (B, H, n_keys=n, n_queries=n).
        print("STAGE 3: TOKEN COLUMNS, CAUSAL ATTENTION, RoPE, KV CACHE")
        print(f"B={B}, n={n}, V={tokenizer.V}, d_emb={d_emb}, H={H}, d_k={attention.d_k}")
        print("Input IDs:", ids.tolist())
        print("Example-zero tokens:", tokens)
        print("Input embeddings:", tuple(x.shape), "; output:", tuple(output.shape))
        for h, head in enumerate(attention.heads):
            print(f"Head {h} weight shapes:", {name: tuple(getattr(head, name).weight.shape)
                                              for name in ("query", "key", "value_down", "value_up")})
        print("\nRoPE FREQUENCIES: radians per token; the same ladder is used by every head")
        for r, frequency in enumerate(attention.rope.inv_freq.tolist()):
            print(f"Features ({2*r}, {2*r+1}): frequency={frequency:.6g}, "
                  f"one full rotation every {2*math.pi/frequency:.2f} tokens")
        print("These are fixed frequencies. Learned Q/K projections let heads use them differently.")
        print("Weights are randomly initialized; these are not learned grammatical patterns yet.")
        print("\nCAUSAL MASK: key rows, query columns; 1 = allowed")
        print(causal_mask(n).int())
        q = attention.heads[0].query(x)  # (B, d_k, n).
        print_token_columns("Head-zero queries before RoPE", q, tokens)
        print_token_columns("Head-zero queries after RoPE", attention.rope(q), tokens)
        print("\nATTENTION PROBABILITIES: example zero, head zero")
        print(probabilities[0, 0])  # (n_keys, n_queries).
        print("Column sums:", probabilities[0, 0].sum(dim=-2))  # (n_queries,).
        print_token_columns("Attention update (before external residual)", output, tokens)

        print("\nCACHE: prefill the first three tokens, then send one new column at a time")
        pieces = [attention(x[..., :3], cache=cache)]  # One (B, d_emb, 3) result.
        print("Prefill: layer=0, processed positions=0..2, cached tokens=", cache.length(0))
        for i in range(3, n):
            next_output, next_probabilities = attention(
                x[..., i:i+1], cache=cache, return_attention=True
            )  # (B, d_emb, 1), (B, H, i+1, 1).
            pieces.append(next_output)
            print(f"Position {i}: queries=1, cached keys={cache.length(0)}, "
                  f"attention shape={tuple(next_probabilities.shape)}")
        cached_output = torch.cat(pieces, dim=-1)  # (B, d_emb, n).
        max_error = (cached_output - output).abs().max().item()  # Scalar.
        torch.testing.assert_close(cached_output, output, rtol=1e-5, atol=1e-7)
        print("Cache layer IDs:", sorted(cache.layers))
        print("Stored rotated keys:", tuple(cache.get(0).keys.shape))
        print("Stored unrotated values:", tuple(cache.get(0).values.shape))
        print(f"Full versus cached maximum absolute difference: {max_error:.3g}")
        print("Full and cached outputs match: True")
        cache.clear()
        print("After clear(), cached tokens:", cache.length(0))
    print("\nOnly attention is demonstrated here. See scripts.inspect_model for the full model; training is available via src.train.")


if __name__ == "__main__":
    main()
