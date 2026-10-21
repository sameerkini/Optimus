"""Top-k sampling and autoregressive generation, with an optional KV cache."""

import math

import torch
from torch import Tensor

from .cache import KVCache
from .model import TinyTransformer


def sample_top_k(logits: Tensor, top_k: int = 10, temperature: float = 1.0,
                 *, generator: torch.Generator | None = None) -> Tensor:
    """Raw logits (B, V) -> one sampled token ID per example (B, 1).

    Retain the k largest scores, normalize among those candidates, then draw.
    k is capped at V; k=1 gives greedy decoding. Temperature must be positive.
    """
    if logits.ndim != 2 or min(logits.shape) <= 0 or not logits.is_floating_point():
        raise ValueError("logits must have nonempty floating-point shape (B, V).")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Sampling logits must be finite.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    k = min(top_k, logits.shape[-1])  # Scalar number of retained candidates.
    candidate_logits, candidate_ids = torch.topk(logits, k, dim=-1)  # Each: (B, k).
    centered = candidate_logits - candidate_logits.amax(dim=-1, keepdim=True)  # (B, k).
    probabilities = torch.softmax(centered / temperature, dim=-1)  # (B, k).
    chosen = torch.multinomial(probabilities, 1, generator=generator)  # (B, 1); candidate index.
    return candidate_ids.gather(dim=-1, index=chosen)  # (B, 1); vocabulary ID.


@torch.no_grad()
def generate(model: TinyTransformer, prefix: Tensor, max_new_tokens: int,
             temperature: float = 1.0, *, top_k: int = 10, use_cache: bool = True,
             generator: torch.Generator | None = None) -> Tensor:
    """Extend IDs (B, n_prefix) -> (B, n_prefix + max_new_tokens).

    Prefill once, then pass only the newly sampled token to the cached model.
    When the context fills, clear and rebuild from the newest n_max tokens.
    This deliberate policy matches uncached cropped-context generation; the
    basic cache does not evict tokens. Full output history is always retained.
    These fixed-length batches have no padding or early EOS stopping yet.
    """
    if prefix.ndim != 2 or min(prefix.shape) <= 0:
        raise ValueError("prefix must have nonempty shape (B, n_prefix).")
    if prefix.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefix must contain int32 or int64 token IDs.")
    if prefix.device != model.token_embedding.weight.device:
        raise ValueError("Prefix and model parameters must be on the same device.")
    if bool(((prefix < 0) | (prefix >= model.config.V)).any()):
        raise ValueError("Prefix token IDs must be between zero and V-1.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    tokens = prefix.clone()  # (B, n_prefix); preserve caller's input tensor.
    if max_new_tokens == 0:
        return tokens
    cache = KVCache(model.config.n_max) if use_cache else None
    was_training = model.training  # Scalar mode flag, restored on return or failure.
    model.eval()
    try:
        for _ in range(max_new_tokens):
            if cache is None:
                context = tokens[:, -model.config.n_max:]  # (B, n_context).
            elif cache.length(0) == 0 or cache.length(0) == model.config.n_max:
                cache.clear()
                context = tokens[:, -model.config.n_max:]  # (B, n_context); prefill/rebuild.
            else:
                context = tokens[:, -1:]  # (B, 1); only the unseen sampled token.
            logits = model(context, cache=cache, last_token_only=True)  # (B, V, 1).
            next_id = sample_top_k(
                logits[..., -1], top_k=top_k, temperature=temperature, generator=generator
            )  # (B, 1); softmax is over the retained vocabulary candidates.
            tokens = torch.cat((tokens, next_id), dim=-1)  # (B, previous_length+1).
        return tokens
    finally:
        model.train(was_training)


def main(argv: list[str] | None = None) -> None:
    """Load a saved model and generate text from the command line."""
    import argparse
    from pathlib import Path

    from .train import load_checkpoint, resolve_device, seed_everything

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    ids = tokenizer.encode(args.prompt)
    if not ids:
        parser.error("--prompt must contain at least one token.")
    if tokenizer.unk_id in ids:
        import sys
        print("Prompt contains tokens outside the training vocabulary (<UNK>).", file=sys.stderr)
    seed_everything(args.seed)
    prefix = torch.tensor([ids], dtype=torch.long, device=device)  # (1, n_prefix).
    output = generate(model, prefix, args.max_new_tokens, args.temperature,
                      top_k=args.top_k, use_cache=args.cache)  # (1, n_prefix + generated).
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
