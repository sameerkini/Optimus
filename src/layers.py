"""Hidden states use (B, d_emb, n): batch, features, tokens.

For one example, x[b] is a (d_emb, n) matrix: each column is one token.
x[b, :, i] is the representation of token position i.

Parameters (embedding tables, weights, biases, gamma, beta) persist across
examples; activations depend on the current input. __init__ creates parameters
once, and calling layer(x) invokes forward(x) through nn.Module.

Text -> Tokenizer.encode -> integer IDs (B, n) -> TokenEmbedding (B, d_emb, n).
The layers inspection script adds learned positions (1, d_emb, n) for comparison.
The current RoPE architecture uses token lookup alone at model input and rotates
queries/keys inside attention instead of adding learned position vectors.
Attention mixes token columns; these layers keep columns independent.
"""

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Project each token column's features: output = weight @ x + bias.

    d_in/d_out are generic feature widths, e.g. d_emb -> d_ff or d_emb -> d_k.
    We store weights as (d_out, d_in), retaining one row per output feature.
    y[b, r, i] = sum_j weight[r, j] * x[b, j, i] + bias[r, 0].
    The sum mixes features j within token i; all examples share the parameters.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("Feature dimensions must be positive.")
        self.d_in = d_in  # Scalar input feature width.
        self.d_out = d_out  # Scalar output feature width.
        self.weight = nn.Parameter(
            torch.randn(d_out, d_in) * 0.02
        )  # (d_out, d_in); small random initial trainable weights.
        self.bias = (
            nn.Parameter(torch.zeros(d_out, 1)) if bias else None
        )  # (d_out, 1) or None; broadcasts over examples and token columns.

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_in, n) -> output: (..., d_out, n)."""
        projected = self.weight @ x  # (d_out, d_in) @ (B, d_in, n) -> (B, d_out, n).
        if self.bias is not None:
            projected = projected + self.bias  # (B, d_out, n).
        return projected


class TokenEmbedding(nn.Module):
    """ID i selects column i of a (d_emb, V) learned table.

    V counts vocabulary entries; n counts token positions in the current input.
    Repeated IDs select the same parameter column, including across documents.
    The tokenizer supplies IDs and V, not the embedding values.
    """

    def __init__(self, V: int, d_emb: int):
        super().__init__()
        if V <= 0 or d_emb <= 0:
            raise ValueError("V and d_emb must be positive.")
        self.d_emb = d_emb  # Scalar model representation width.
        self.weight = nn.Parameter(
            torch.randn(d_emb, V) * 0.02
        )  # (d_emb, V); one learned column vector per token ID.

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: (B, n) integers -> embeddings: (B, d_emb, n)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, n).")
        B, n = token_ids.shape  # Scalars: examples in batch, tokens in each example.
        # Flattening is temporary bookkeeping; it does not mix examples.
        flat_ids = token_ids.reshape(-1)  # (B*n,).
        selected_columns = self.weight.index_select(1, flat_ids)  # (d_emb, B*n).
        grouped = selected_columns.reshape(self.d_emb, B, n)  # (d_emb, B, n).
        embeddings = grouped.permute(1, 0, 2)  # (B, d_emb, n); put batch first.
        return embeddings


class PositionalEmbedding(nn.Module):
    """Optional learned-position baseline; do not add this in the RoPE model.

    Return (1, d_emb, n) so position columns broadcast over examples B.
    E[:, token_id] + P[:, position] retains d_emb features.
    """

    def __init__(self, n_max: int, d_emb: int):
        super().__init__()
        if n_max <= 0 or d_emb <= 0:
            raise ValueError("n_max and d_emb must be positive.")
        self.n_max = n_max  # Scalar cap; actual input length is n <= n_max.
        self.weight = nn.Parameter(
            torch.randn(d_emb, n_max) * 0.02
        )  # (d_emb, n_max); column i represents position i.

    def forward(self, n: int) -> Tensor:
        """Return positions (1, d_emb, n), shared across the batch."""
        if not 1 <= n <= self.n_max:
            raise ValueError("n must be between 1 and n_max.")
        positions = self.weight[:, :n]  # (d_emb, n).
        return positions.unsqueeze(0)  # (1, d_emb, n).


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent feature pairs of Q or K; values are left unchanged.

    For pair r at token position p, angle = p / base**(2*r/d_k).
    [a, b] becomes [a*cos(angle)-b*sin(angle), a*sin(angle)+b*cos(angle)].
    Frequencies are fixed buffers, not trainable parameters. See RoFormer §3.2:
    https://arxiv.org/html/2104.09864v5#S3.SS2
    """

    inv_freq: Tensor  # (d_k/2,); type hint for the buffer registered below.

    def __init__(self, d_k: int, base: float = 10000.0):
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("RoPE requires a positive even d_k for feature pairs.")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("RoPE base must be finite and greater than one.")
        self.d_k = d_k  # Scalar rotated feature width per head.
        pair_features = torch.arange(0, d_k, 2, dtype=torch.float32)  # (d_k/2,): 0, 2, 4, ...
        self.register_buffer(
            "inv_freq", base ** (-pair_features / d_k)
        )  # (d_k/2,); each adjacent feature pair gets a different angular frequency.
        # Every head uses this frequency ladder; its learned Q/K projections differ.

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """x: (..., d_k, n); token positions are start_pos through start_pos+n-1."""
        if x.ndim < 2 or x.shape[-2] != self.d_k or not x.is_floating_point():
            raise ValueError("RoPE expects floating-point (..., d_k, n) input.")
        if start_pos < 0:
            raise ValueError("start_pos must be nonnegative.")
        n = x.shape[-1]  # Scalar number of new token columns.
        angle_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        positions = torch.arange(
            start_pos, start_pos + n, device=x.device, dtype=angle_dtype
        )  # (n,); absolute logical positions, including the cached prefix.
        angles = self.inv_freq.to(device=x.device, dtype=angle_dtype)[:, None] * positions
        # angles: (d_k/2, n); broadcast over B and H below.
        cosine = angles.cos().to(dtype=x.dtype)  # (d_k/2, n).
        sine = angles.sin().to(dtype=x.dtype)  # (d_k/2, n).
        even = x[..., 0::2, :]  # (..., d_k/2, n).
        odd = x[..., 1::2, :]  # (..., d_k/2, n).
        rotated_even = even * cosine - odd * sine  # (..., d_k/2, n).
        rotated_odd = even * sine + odd * cosine  # (..., d_k/2, n).
        pairs = torch.stack((rotated_even, rotated_odd), dim=-2)  # (..., d_k/2, 2, n).
        return pairs.flatten(-3, -2)  # (..., d_k, n); restore adjacent pairs.


class LayerNorm(nn.Module):
    """Normalize each token's features independently of other tokens/examples.

    x: (B, d_emb, n). dim=-2 selects features; keepdim=True retains (B, 1, n).
    gamma/beta have one trainable value per feature, shared across B and n.
    """

    def __init__(self, d_emb: int, eps: float = 1e-5):
        super().__init__()
        if d_emb <= 0 or eps <= 0:
            raise ValueError("d_emb and epsilon must be positive.")
        self.eps = eps  # Scalar epsilon for numerical stability.
        self.gamma = nn.Parameter(torch.ones(d_emb, 1))  # (d_emb, 1).
        self.beta = nn.Parameter(torch.zeros(d_emb, 1))  # (d_emb, 1).

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., d_emb, n) -> output: (..., d_emb, n)."""
        mean = x.mean(dim=-2, keepdim=True)  # (B, 1, n); one mean per token column.
        centered = x - mean  # (B, d_emb, n); broadcast each token's own mean.
        variance = (centered * centered).mean(
            dim=-2, keepdim=True
        )  # (B, 1, n); population variance, dividing by d_emb.
        normalized = centered / torch.sqrt(variance + self.eps)  # (B, d_emb, n).
        output = self.gamma * normalized + self.beta  # (B, d_emb, n).
        return output


class FeedForward(nn.Module):
    """Two projections and GELU, independently at each token column.

    (B, d_emb, n) -> (B, d_ff, n) -> (B, d_emb, n); default d_ff = 4*d_emb.
    Expansion adds features, not token positions. Norm/residual are external.
    """

    def __init__(self, d_emb: int, d_ff: int | None = None):
        super().__init__()
        d_ff = 4 * d_emb if d_ff is None else d_ff  # Scalar MLP hidden width.
        self.up = Linear(d_emb, d_ff)  # W1: (d_ff, d_emb); b1: (d_ff, 1).
        self.down = Linear(d_ff, d_emb)  # W2: (d_emb, d_ff); b2: (d_emb, 1).

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, d_emb, n) -> output: (B, d_emb, n)."""
        hidden = self.up(x)  # (B, d_ff, n).
        activated = 0.5 * hidden * (
            1.0 + torch.erf(hidden / math.sqrt(2.0))
        )  # (B, d_ff, n); exact elementwise GELU introduces nonlinearity.
        output = self.down(activated)  # (B, d_emb, n).
        return output
