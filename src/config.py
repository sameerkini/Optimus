"""Small starting dimensions, not tuned hyperparameters."""

from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    V: int  # Vocabulary size, including the reserved unknown token.
    n_max: int = 64  # Maximum input sequence length; actual n may be smaller.
    d_emb: int = 64  # Features in each token/model representation.
    H: int = 4  # Number of attention heads.
    L: int = 2  # Complete blocks: LN -> attention -> LN -> MLP, with residuals.
    d_ff: int | None = None  # Defaults to 4*d_emb, including when d_emb changes.
    dropout: float = 0.0  # Start deterministic; add regularization later.
    rope_base: float = 10000.0  # Fixed RoPE frequency base; no additive position table.
    tie_embeddings: bool = True  # Unembedding uses the transpose of the input table.

    def __post_init__(self) -> None:
        if min(self.V, self.n_max, self.d_emb, self.H, self.L) <= 0:
            raise ValueError("V, n_max, d_emb, H, and L must be positive.")
        if self.d_emb % self.H != 0:
            raise ValueError("d_emb must be divisible by H so d_k is an integer.")
        if self.d_k % 2:
            raise ValueError("RoPE requires even d_k so features form rotation pairs.")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and between zero (inclusive) and one.")
        if self.d_ff is None:
            self.d_ff = 4 * self.d_emb
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")

    @property
    def d_k(self) -> int:
        """Query/key features per head; values use this same width here."""
        return self.d_emb // self.H


@dataclass
class TrainConfig:
    B: int = 16
    n: int = 64  # Input window length, <= model n_max.
    learning_rate: float = 3e-4
    max_steps: int = 200  # Small initial budget, not a convergence guarantee.
    weight_decay: float = 0.01
    grad_clip: float = 1.0  # Maximum global gradient norm.
    log_every: int = 10
    eval_every: int = 50
    eval_batches: int | None = None  # None evaluates all validation windows.
    patience: int = 5  # Validation checks without improvement; zero disables stopping.
    overfit_one_batch: bool = False  # Reuse one batch to debug optimization.
    num_threads: int | None = None  # Optional CPU thread cap for timing small models.
    seed: int = 42
    device: str = "cpu"  # cpu, mps, cuda, or auto; start with CPU correctness checks.

    def __post_init__(self) -> None:
        if min(self.B, self.n, self.max_steps, self.log_every, self.eval_every) <= 0:
            raise ValueError("B, n, max_steps, log_every, and eval_every must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be finite and positive.")
        if self.patience < 0 or (self.eval_batches is not None and self.eval_batches <= 0):
            raise ValueError("patience must be nonnegative; eval_batches must be positive.")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
