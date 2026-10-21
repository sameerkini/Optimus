"""Causal multi-head attention, RoPE, and optional per-layer KV caching.

Attention = V * softmax(K^T Q / sqrt(d_k)); we treat each token vector as a column

X: (B, d_emb, n). Q/K/values: (B, H, d_k, n), d_k = d_emb // H.
Scores = K.transpose(-2, -1) @ Q: (B, H, n_keys, n_queries).
Softmax runs down key rows; values @ probabilities returns token columns.
Each head maps back to d_emb using value_up; sum the H head updates.
Normalization and residual addition belong outside this module.
"""

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .cache import KVCache
from .layers import Linear, RotaryPositionalEmbedding


def causal_mask(n: int, device: str | torch.device = "cpu", *, start_pos: int = 0) -> Tensor:
    """True = allowed. Shape (start_pos+n, n): key rows, new-query columns.

    Block a key at position i when i > query position j. At start_pos=0,
    the allowed region is upper triangular, including the diagonal.
    A cached prefix makes a rectangular mask: queries start at start_pos.
    """
    if n <= 0 or start_pos < 0:
        raise ValueError("n must be positive and start_pos must be nonnegative.")
    key_positions = torch.arange(start_pos + n, device=device)  # (n_keys,).
    query_positions = torch.arange(start_pos, start_pos + n, device=device)  # (n,).
    return key_positions[:, None] <= query_positions[None, :]  # (n_keys, n).


class AttentionHead(nn.Module):
    """One head's four learned projections, with column biases."""

    def __init__(self, d_emb: int, d_k: int):
        super().__init__()
        self.query = Linear(d_emb, d_k)  # W_Q: (d_k, d_emb); bias: (d_k, 1).
        self.key = Linear(d_emb, d_k)  # W_K: (d_k, d_emb); bias: (d_k, 1).
        self.value_down = Linear(d_emb, d_k)  # (d_k, d_emb); bias: (d_k, 1).
        self.value_up = Linear(d_k, d_emb)  # (d_emb, d_k); bias: (d_emb, 1).

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: (B, d_emb, n) -> three tensors of shape (B, d_k, n)."""
        return self.query(x), self.key(x), self.value_down(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_emb: int, H: int, *, layer_idx: int, rope_base: float = 10000.0):
        super().__init__()
        if d_emb <= 0 or H <= 0 or d_emb % H:
            raise ValueError("d_emb and H must be positive, with d_emb divisible by H.")
        if layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative and unique within a model.")
        self.d_emb = d_emb  # Scalar model feature width.
        self.H = H  # Scalar number of independent heads.
        self.d_k = d_emb // H  # Scalar query/key/value feature width per head.
        self.layer_idx = layer_idx  # Scalar cache slot; block ell uses layer_idx=ell.
        self.rope = RotaryPositionalEmbedding(self.d_k, base=rope_base)
        # ModuleList registers parameters; this hint tells the editor every item is an AttentionHead.
        self.heads = cast(
            Sequence[AttentionHead],
            nn.ModuleList(AttentionHead(d_emb, self.d_k) for _ in range(H)),
        )  # H registered heads; cast changes typing only, not the runtime container.

    def forward(self, x: Tensor, allowed: Tensor | None = None, *, cache: KVCache | None = None,
                return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """(B, d_emb, n_new) -> (B, d_emb, n_new), preserving token columns.

        Without cache: process a full sequence, including during training.
        With cache: eval() + torch.no_grad(); send ONLY unseen token columns.
        One shared cache holds all layers, each with a distinct layer_idx.
        allowed optionally restricts the causal mask: bool (n_keys, n_new),
        shared over B and H. It cannot permit future keys or empty a query column.
        return_attention adds probabilities (B, H, n_keys, n_new) for inspection.
        """
        if x.ndim != 3 or x.shape[1] != self.d_emb or x.shape[0] == 0 or x.shape[-1] == 0:
            raise ValueError("x must have nonempty shape (B, d_emb, n_new).")
        if cache is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("Use KV caching only in eval mode inside torch.no_grad().")
        start_pos = 0 if cache is None else cache.length(self.layer_idx)  # Scalar past length.
        n_new = x.shape[-1]  # Scalar new-query count; decode usually uses one.
        if cache is not None and start_pos + n_new > cache.n_max:
            raise ValueError("KV cache would exceed n_max; clear and rebuild the context.")
        legal = causal_mask(n_new, x.device, start_pos=start_pos)  # (n_keys, n_new).
        if allowed is not None:
            if allowed.dtype != torch.bool or allowed.shape != legal.shape:
                raise ValueError("allowed must be boolean with shape (n_keys, n_new).")
            legal = legal & allowed.to(device=x.device)  # (n_keys, n_new); retain causality.
        if not bool(legal.any(dim=-2).all()):
            raise ValueError("Each query must have at least one allowed key.")

        projected = [head.project(x) for head in self.heads]  # H tuples of (B, d_k, n_new).
        queries = torch.stack([item[0] for item in projected], dim=1)  # (B, H, d_k, n_new).
        keys = torch.stack([item[1] for item in projected], dim=1)  # (B, H, d_k, n_new).
        values = torch.stack([item[2] for item in projected], dim=1)  # (B, H, d_k, n_new).
        queries = self.rope(queries, start_pos=start_pos)  # (B, H, d_k, n_new).
        keys = self.rope(keys, start_pos=start_pos)  # (B, H, d_k, n_new); rotate once.
        if cache is not None:
            history = cache.append(self.layer_idx, keys, values)
            keys, values = history.keys, history.values  # Each: (B, H, d_k, n_keys).
        scores = keys.transpose(-2, -1) @ queries / math.sqrt(self.d_k)
        # scores: (B, H, n_keys, n_new); each column belongs to one query.
        masked_scores = scores.masked_fill(~legal, float("-inf"))  # (B, H, n_keys, n_new).
        probabilities = torch.softmax(masked_scores, dim=-2)  # (B, H, n_keys, n_new).
        head_outputs = values @ probabilities  # (B, H, d_k, n_new).
        updates = torch.stack(
            [head.value_up(head_outputs[:, h]) for h, head in enumerate(self.heads)], dim=1
        )  # (B, H, d_emb, n_new); each head returns an embedding-space update.
        output = updates.sum(dim=1)  # (B, d_emb, n_new); add the heads' contributions.
        return (output, probabilities) if return_attention else output
