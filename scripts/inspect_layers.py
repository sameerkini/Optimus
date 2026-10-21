"""Trace real text through stage 2 using (B, d_emb, n).

Run from the project folder: python -m scripts.inspect_layers
"""

import json
import math
from pathlib import Path

import torch

from src.layers import FeedForward, LayerNorm, PositionalEmbedding, TokenEmbedding
from src.tokenizer import Tokenizer


def print_token_columns(name: str, values: torch.Tensor, tokens: list[str], example: int = 0) -> None:
    """Display values[example]: (features, n), with one token per column."""
    matrix = values[example].detach()  # (number_of_features, n).
    print(f"\n{name}: shape={tuple(values.shape)}; displaying example {example}")
    print("Columns are tokens; rows are features (or one statistic per token for LN).")
    token_labels = [f"i={position} {token!r}" for position, token in enumerate(tokens)]
    print(f"{'feature / token':>20}" + "".join(f"{label:>15}" for label in token_labels))
    for feature, row in enumerate(matrix.tolist()):
        print(f"{f'feature {feature}':>20}" + "".join(f"{value:15.5f}" for value in row))


def main() -> None:
    torch.manual_seed(42)
    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "gutenberg"
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    training_paths = sorted(corpus_dir / item["path"] for item in manifest["documents"]
                            if item["split"] == "train")
    tokenizer = Tokenizer.from_files(training_paths)
    first_book = training_paths[0].stem.split("_")[0]
    other_book_path = next(path for path in training_paths if path.stem.split("_")[0] != first_book)
    example_paths = [training_paths[0], other_book_path]  # B=2 source paths.
    texts = []  # B inspection samples, not the full corpus.
    for path in example_paths:
        document = path.read_text(encoding="utf-8")
        texts.append(document.split("\n\n", 1)[1].lstrip())
    B, n, d_emb = 2, 6, 4  # Scalars; small dimensions make every value readable.
    token_strings = [tokenizer.tokenize(text)[:n] for text in texts]  # B lists of n strings.
    token_ids = torch.tensor(
        [tokenizer.encode(text)[:n] for text in texts], dtype=torch.long
    )  # (B=2, n=6): integer IDs, not embedding coordinates.

    embedding = TokenEmbedding(V=tokenizer.V, d_emb=d_emb)  # E: (4, V).
    positional = PositionalEmbedding(n_max=64, d_emb=d_emb)  # P: (4, 64).
    norm = LayerNorm(d_emb)  # gamma and beta: (4, 1) each.
    ffn = FeedForward(d_emb)  # W1: (16, 4); W2: (4, 16).

    token_vectors = embedding(token_ids)  # (2, 4, 6).
    position_vectors = positional(n)  # (1, 4, 6), shared across B.
    x = token_vectors + position_vectors  # (2, 4, 6).
    means = x.mean(dim=-2, keepdim=True)  # (2, 1, 6): one mean per token column.
    centered = x - means  # (2, 4, 6).
    variances = centered.square().mean(dim=-2, keepdim=True)  # (2, 1, 6).
    normalized = norm(x)  # (2, 4, 6).
    hidden = ffn.up(normalized)  # (2, 16, 6): more features, still six tokens.
    activated = 0.5 * hidden * (
        1.0 + torch.erf(hidden / math.sqrt(2.0))
    )  # (2, 16, 6).
    ffn_update = ffn(normalized)  # (2, 4, 6).
    after_residual = x + ffn_update  # (2, 4, 6): one pre-LN FFN sublayer.
    torch.testing.assert_close(ffn.down(activated), ffn_update)

    print("LAYOUT: (B, d_emb, n) = (examples, features, token positions)")
    print("Stage 2 learned-position comparison; stage 3 uses RoPE without additive positions.")
    print(f"Vocabulary fitted on {len(training_paths)} training documents; V={tokenizer.V:,}.")
    print(f"Inspection dimensions: B={B}, n={n}, d_emb={d_emb}; model cap n_max=64.")
    print("B counts this batch's example chunks, not the number of corpus documents.")
    for index, path in enumerate(example_paths):
        print(f"\nExample {index}, source {path.name}:")
        print("Text prefix:", repr(texts[index][:100]))
        print("Tokens:", token_strings[index])
        print("IDs:", token_ids[index].tolist())

    print("\nSHAPES")
    print("Embedding table E:", tuple(embedding.weight.shape), "= (d_emb, V)")
    print("Position table P:", tuple(positional.weight.shape), "= (d_emb, n_max)")
    print("Token IDs:", tuple(token_ids.shape), "= (B, n)")
    print("Selected embeddings:", tuple(token_vectors.shape), "= (B, d_emb, n)")
    print("Lookup: IDs (2, 6) -> flat IDs (12,) -> columns (4, 12) -> (4, 2, 6) -> (2, 4, 6)")
    print("Selected positions:", tuple(position_vectors.shape), "= (1, d_emb, n)")
    print("LayerNorm means/variances:", tuple(means.shape), "= (B, 1, n)")
    print("LayerNorm gamma/beta:", tuple(norm.gamma.shape), "= (d_emb, 1)")
    print("FFN stored W1:", tuple(ffn.up.weight.shape), "= (d_ff, d_emb)")
    print("FFN stored W2:", tuple(ffn.down.weight.shape), "= (d_emb, d_ff)")
    print("FFN column biases:", tuple(ffn.up.bias.shape), tuple(ffn.down.bias.shape))
    print("FFN expanded features:", tuple(hidden.shape), "= (B, d_ff, n)")
    print("Residual result:", tuple(after_residual.shape), "= (B, d_emb, n)")

    first_id = token_ids[0, 0].item()  # Scalar ID.
    lookup_column = embedding.weight[:, first_id:first_id + 1]  # (d_emb, 1).
    selected_column = token_vectors[0, :, :1]  # (d_emb, 1).
    print(f"\nEmbedding column for {token_strings[0][0]!r}, ID {first_id}:", lookup_column.detach())
    print("Lookup equals selected table column:", torch.equal(lookup_column, selected_column))

    print_token_columns("Token embeddings, first example", token_vectors, token_strings[0])
    print_token_columns("Token embeddings, second example", token_vectors, token_strings[1], example=1)
    print_token_columns("Learned positions", position_vectors, token_strings[0])
    print_token_columns("X = token embeddings + positions", x, token_strings[0])
    print_token_columns("LN means", means, token_strings[0])
    print_token_columns("LN centered features", centered, token_strings[0])
    print_token_columns("LN population variances", variances, token_strings[0])
    print_token_columns("LN output", normalized, token_strings[0])
    print_token_columns("FFN expanded features before GELU", hidden, token_strings[0])
    print_token_columns("FFN after GELU", activated, token_strings[0])
    print_token_columns("FFN update", ffn_update, token_strings[0])
    print_token_columns("X + FFN(LN(X))", after_residual, token_strings[0])

    # Both layouts contain the same values and the same per-example 2D matrices.
    batch_last = x.permute(1, 2, 0)  # (d_emb, n, B); a view with reordered axes.
    print("\nWHY BATCH FIRST?")
    print("X[b] shape:", tuple(x[0].shape), "= (d_emb, n)")
    print("Batch-last view shape:", tuple(batch_last.shape), "= (d_emb, n, B)")
    print("batch_last[:, :, b] shape:", tuple(batch_last[:, :, 0].shape))
    print("Both example slices are identical:", torch.equal(x[0], batch_last[:, :, 0]))
    print("Batch-first: W (d_ff, d_emb) @ X (B, d_emb, n) -> (B, d_ff, n).")
    print("Matmul uses the last two axes as a matrix; batch-last needs another axis reorder.")

    changed_ids = token_ids.clone()  # (B, n).
    changed_ids[0, 0] = token_ids[1, 0]  # Change one token ID in one example.
    assert changed_ids[0, 0] != token_ids[0, 0]
    changed_x = embedding(changed_ids) + position_vectors  # (B, d_emb, n).
    changed_output = changed_x + ffn(norm(changed_x))  # (B, d_emb, n).
    maximum_changes = (changed_output - after_residual).abs().amax(dim=-2)  # (B, n).
    print("\nCROSS-TALK CHECK: maximum feature change per token:")
    print(maximum_changes.detach())
    other_tokens_unchanged = torch.allclose(changed_output[0, :, 1:], after_residual[0, :, 1:])
    other_example_unchanged = torch.allclose(changed_output[1], after_residual[1])
    print("Other token positions unchanged:", other_tokens_unchanged)
    print("Other batch example unchanged:", other_example_unchanged)
    assert other_tokens_unchanged and other_example_unchanged

    toy_loss = after_residual.square().mean()  # Scalar diagnostic, not a language-model objective.
    toy_loss.backward()
    print("\nAUTOGRAD CHECK (no optimizer step)")
    for module_name, module in (("embedding", embedding), ("position", positional),
                                ("norm", norm), ("ffn", ffn)):
        for name, parameter in module.named_parameters():
            finite = parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            print(f"{module_name}.{name}: shape={tuple(parameter.shape)}, finite gradient={finite}")
            assert finite
    print("\nThis is stage 2 inspection. Inspect the assembled model with scripts.inspect_model; training is available via src.train.")


if __name__ == "__main__":
    main()
