"""Inference KV storage, indexed by attention-layer number.

One cache belongs to one batch of sequences. Batch order stays fixed, and all
examples advance together. Clear it before a new prompt or after changing weights.
This first cache appends ordinary tensors; it does not implement memory paging.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LayerKV:
    keys: Tensor  # (B, H, d_k, n_cached); already rotated by RoPE.
    values: Tensor  # (B, H, d_k, n_cached); not rotated.

    @property
    def n_tokens(self) -> int:
        return self.keys.shape[-1]


class KVCache:
    """Keep separate K/V history for each layer_idx, starting at position zero."""

    def __init__(self, n_max: int):
        if n_max <= 0:
            raise ValueError("n_max must be positive.")
        self.n_max = n_max  # Scalar context limit; no automatic eviction.
        self.layers: dict[int, LayerKV] = {}  # Layer number -> its K/V history.

    def get(self, layer_idx: int) -> LayerKV | None:
        if layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative.")
        return self.layers.get(layer_idx)

    def length(self, layer_idx: int) -> int:
        entry = self.get(layer_idx)
        return 0 if entry is None else entry.n_tokens

    def append(self, layer_idx: int, keys: Tensor, values: Tensor) -> LayerKV:
        """Append new columns (B, H, d_k, n_new); return the whole layer history.

        Cache entries have no autograd graph. The attention module restricts
        cache use to eval mode with no gradients.
        """
        old = self.get(layer_idx)
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[-1] == 0:
            raise ValueError("keys and values must have matching (B, H, d_k, n_new) shapes.")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("keys and values must share a device and dtype.")
        if self.length(layer_idx) + keys.shape[-1] > self.n_max:
            raise ValueError("KV cache would exceed n_max; clear and rebuild the context.")
        keys = keys.detach()  # (B, H, d_k, n_new); inference activations only.
        values = values.detach()  # (B, H, d_k, n_new).
        if old is not None:
            if old.keys.shape[:-1] != keys.shape[:-1]:
                raise ValueError("Cached B, H, and d_k must match; clear for a new batch.")
            if old.keys.device != keys.device or old.keys.dtype != keys.dtype:
                raise ValueError("Cached device/dtype must match; clear after moving the model.")
            keys = torch.cat((old.keys, keys), dim=-1)  # (B, H, d_k, n_total).
            values = torch.cat((old.values, values), dim=-1)  # (B, H, d_k, n_total).
        entry = LayerKV(keys, values)
        self.layers[layer_idx] = entry
        return entry

    def clear(self) -> None:
        """Start a new batch of prompts; discard every layer's history."""
        self.layers.clear()
