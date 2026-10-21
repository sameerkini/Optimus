"""Print real-token alignment, model shapes, tied weights, and cached generation.

Run: .venv/bin/python -m scripts.inspect_model
No training or optimizer updates are performed.
"""

import json
from pathlib import Path

import torch

from src.cache import KVCache
from src.config import ModelConfig
from src.generate import generate
from src.model import TinyTransformer
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
             for path in (paths[0], other)]  # B=2 source samples.
    B, n = 2, 6  # Scalars: example count and input-token count.
    spans = torch.tensor([tokenizer.encode(text)[:n+1] for text in texts])  # (B, n+1).
    inputs, targets = spans[:, :-1], spans[:, 1:]  # Each: (B, n), one-token offset.
    config = ModelConfig(V=tokenizer.V, d_emb=8, H=2, L=2, n_max=16)
    model = TinyTransformer(config).eval()

    print("STAGE 4: PRE-LN MODEL, TIED UNEMBEDDING, AND CACHED GENERATION")
    print(f"B={B}, n={n}, V={config.V}, d_emb={config.d_emb}, H={config.H}, "
          f"d_k={config.d_k}, d_ff={config.d_ff}, L={config.L}, n_max={config.n_max}")
    print("\nONE SPAN, MULTIPLE TRAINING PREDICTIONS (example zero)")
    pieces = tokenizer.tokenize(texts[0])[:n+1]  # n+1 displayed token strings.
    for i in range(n):
        print(f"Position {i}: input={pieces[i]!r}, target={pieces[i+1]!r}; "
              f"visible input positions=0..{i}")
    print("Input IDs:", inputs.tolist())
    print("Target IDs:", targets.tolist())
    print("Whitespace and punctuation count as tokens in this tokenizer.")

    with torch.no_grad():
        logits = model(inputs)  # (B, V, n); training uses all n columns.
        last_logits = model(inputs, last_token_only=True)  # (B, V, 1).
        torch.testing.assert_close(last_logits, logits[..., -1:])
        print("\nARCHITECTURE")
        print("IDs:", tuple(inputs.shape), "-> embeddings:", tuple(model.token_embedding(inputs).shape))
        print("Each block: LN -> attention -> residual add -> LN -> MLP -> residual add")
        print("Layer indices:", [block.attention.layer_idx for block in model.blocks])
        print("After L blocks and final LN: (B, d_emb, n)")
        print("E:", tuple(model.token_embedding.weight.shape))
        print("Unembedding E.T:", tuple(model.unembedding_weight.shape))
        print("Same underlying parameter storage:",
              model.unembedding_weight.data_ptr() == model.token_embedding.weight.data_ptr())
        print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
        print("Parameters saved by weight sharing:", config.V * config.d_emb)
        print("All-position raw logits:", tuple(logits.shape))
        print("Generation-only last-position logits:", tuple(last_logits.shape))

        mlp_lengths = [[] for _ in model.blocks]  # L lists of observed n_new values.
        hooks = [block.ffn.register_forward_pre_hook(
            lambda module, values, ell=ell: mlp_lengths[ell].append(values[0].shape[-1]))
                 for ell, block in enumerate(model.blocks)]
        cache = KVCache(config.n_max)
        outputs = [model(inputs[:, :3], cache=cache)]  # One (B, V, 3) result.
        for i in range(3, n):
            outputs.append(model(inputs[:, i:i+1], cache=cache))  # Each: (B, V, 1).
        cached_logits = torch.cat(outputs, dim=-1)  # (B, V, n).
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(cached_logits, logits, atol=1e-6, rtol=1e-5)
        print("\nKV CACHE AND MLP WORK")
        for ell, lengths in enumerate(mlp_lengths):
            print(f"Layer {ell}: MLP token counts per call={lengths}; "
                  f"cached tokens={cache.length(ell)}; K shape={tuple(cache.get(ell).keys.shape)}")
        print("Full/cached maximum logit difference:", (cached_logits-logits).abs().max().item())
        print("Full and cached model logits agree: True")

    print("\nTOP-K GENERATION MECHANICS (RANDOM, UNTRAINED WEIGHTS)")
    prefix = inputs[:1, :3]  # (B=1, n_prefix=3).
    sampled = generate(model, prefix, max_new_tokens=4, top_k=5,
                       generator=torch.Generator().manual_seed(7))  # (1, 7).
    print("Prefix:", repr(tokenizer.decode(prefix[0].tolist())))
    print("Sampled text:", repr(tokenizer.decode(sampled[0].tolist())))
    print("This demonstrates sampling, not learned language quality.")
    print("This inspection did not train weights. The training entry point is python -m src.train.")


if __name__ == "__main__":
    main()
