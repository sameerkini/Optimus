"""A pre-LN decoder with token columns (B, d_emb, n).

L counts complete blocks; each block has two residual sublayers with separately learned norms:
    y = x + attention(norm_attention(x))
    z = y + ffn(norm_ffn(y))
Each intermediate above has shape (B, d_emb, n). Add final LN after all blocks.
Use token embeddings without additive positions: attention applies RoPE to Q/K.
The same optional KVCache goes through all blocks, indexed by layer_idx=ell.
Training returns raw logits at every position. Generation can project only the
last position; softmax and sampling live in generate.py, outside the model.
"""

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .attention import CausalSelfAttention
from .cache import KVCache
from .config import ModelConfig
from .layers import FeedForward, LayerNorm, Linear, TokenEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, *, layer_idx: int):
        super().__init__()
        self.norm_attention = LayerNorm(config.d_emb)  # gamma/beta: (d_emb, 1).
        self.attention = CausalSelfAttention(
            config.d_emb, config.H, layer_idx=layer_idx, rope_base=config.rope_base
        )  # (B, d_emb, n_new) -> (B, d_emb, n_new).
        self.norm_ffn = LayerNorm(config.d_emb)  # Separate learned norm parameters.
        self.ffn = FeedForward(config.d_emb, config.d_ff)  # d_emb -> d_ff -> d_emb.
        self.dropout_attention = nn.Dropout(config.dropout)  # Default p=0: identity.
        self.dropout_ffn = nn.Dropout(config.dropout)  # No dimension changes.

    def forward(self, x: Tensor, *, cache: KVCache | None = None) -> Tensor:
        """Preserve (B, d_emb, n_new); cached decoding normally has n_new=1."""
        normalized = self.norm_attention(x)  # (B, d_emb, n_new).
        attention_update = self.attention(normalized, cache=cache)  # (B, d_emb, n_new).
        after_attention = x + self.dropout_attention(attention_update)  # (B, d_emb, n_new).
        normalized = self.norm_ffn(after_attention)  # (B, d_emb, n_new).
        ffn_update = self.ffn(normalized)  # (B, d_emb, n_new); only new columns.
        return after_attention + self.dropout_ffn(ffn_update)  # (B, d_emb, n_new).


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.V, config.d_emb)  # E: (d_emb, V).
        self.blocks = cast(
            Sequence[TransformerBlock],
            nn.ModuleList(TransformerBlock(config, layer_idx=ell) for ell in range(config.L)),
        )  # L registered blocks, with a concrete element type for the editor.
        self.final_norm = LayerNorm(config.d_emb)  # gamma/beta: (d_emb, 1).
        self.unembedding = (
            None if config.tie_embeddings else Linear(config.d_emb, config.V, bias=False)
        )  # Optional independent weight: (V, d_emb).
        self.output_bias = nn.Parameter(torch.zeros(config.V, 1))  # (V, 1).

    @property
    def unembedding_weight(self) -> Tensor:
        """(V, d_emb): a transpose view of E when tied, not a parameter copy."""
        if self.unembedding is None:
            return self.token_embedding.weight.T  # (V, d_emb); gradients also update E.
        return self.unembedding.weight  # (V, d_emb); learned independently.

    def forward(self, token_ids: Tensor, *, cache: KVCache | None = None,
                last_token_only: bool = False) -> Tensor:
        """IDs (B, n_new) -> raw logits (B, V, n_new), or (B, V, 1).

        Without cache, n_new is the full input length. With cache, supply only
        unseen tokens. Each output column predicts the NEXT token, not itself.
        last_token_only saves vocabulary-projection work during prompt prefill;
        all prompt tokens must still pass through the blocks to build their K/V.
        """
        if token_ids.ndim != 2 or min(token_ids.shape) <= 0:
            raise ValueError("token_ids must have nonempty shape (B, n_new).")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must contain int32 or int64 token IDs.")
        if token_ids.device != self.token_embedding.weight.device:
            raise ValueError("Token IDs and model parameters must be on the same device.")
        if bool(((token_ids < 0) | (token_ids >= self.config.V)).any()):
            raise ValueError("Token IDs must be between zero and V-1.")
        B, n_new = token_ids.shape  # Scalar batch size and unseen-token count.
        n_cached = 0  # Scalar prefix length; zero during ordinary training.
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Use KV caching only in eval mode inside torch.no_grad().")
            if set(cache.layers) - set(range(self.config.L)):
                raise ValueError("Cache contains layer indices outside this model; clear it.")
            lengths = [cache.length(ell) for ell in range(self.config.L)]  # L scalar lengths.
            if len(set(lengths)) != 1:
                raise ValueError("All model layers must have the same cached prefix length.")
            n_cached = lengths[0]
            for entry in cache.layers.values():
                if entry.keys.shape[:3] != (B, self.config.H, self.config.d_k):
                    raise ValueError("Cached B, H, and d_k must match; clear for a new batch.")
                if entry.keys.device != token_ids.device:
                    raise ValueError("Cached device must match; clear after moving the model.")
        if n_cached + n_new > self.config.n_max:
            raise ValueError("Input plus cached prefix exceeds the model's n_max.")
        if cache is not None and n_cached + n_new > cache.n_max:
            raise ValueError("Input plus cached prefix exceeds the cache's n_max.")

        x = self.token_embedding(token_ids)  # (B, d_emb, n_new); no additive positions.
        for block in self.blocks:
            x = block(x, cache=cache)  # (B, d_emb, n_new); reuse each layer's own K/V.
        hidden = self.final_norm(x)  # (B, d_emb, n_new).
        if last_token_only:
            hidden = hidden[..., -1:]  # (B, d_emb, 1); retain the column axis.
        logits = self.unembedding_weight @ hidden + self.output_bias  # (B, V, n_out).
        return logits  # n_out = 1 when last_token_only, otherwise n_new; no softmax here.
