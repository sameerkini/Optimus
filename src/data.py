"""Document splits and batches. No optimization or model code belongs here."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

IGNORE_INDEX = -100  # Padding targets contribute no cross-entropy loss.


@dataclass(frozen=True)
class Document:
    path: str  # Manifest-relative name, retained in run/checkpoint metadata.
    split: str  # train, val, or test.
    text: str
    sha256: str


def load_splits(manifest_path: str | Path) -> dict[str, list[Document]]:
    """Read and verify PREASSIGNED document splits before fitting the tokenizer.

    Accept val/validation as equivalent labels. Reject repeated paths, duplicate
    document contents, changed files, and missing splits. Never reshuffle words.
    """
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, list[Document]] = {name: [] for name in ("train", "val", "test")}
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for item in manifest["documents"]:
        split = "val" if item["split"] == "validation" else item["split"]
        if split not in splits:
            raise ValueError(f"Unknown document split: {split!r}.")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or path in seen_paths:
            raise ValueError("Document paths must be unique and inside the corpus folder.")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Document hash mismatch: {item['path']}.")
        if digest in seen_hashes:
            raise ValueError("Duplicate document contents could leak across splits.")
        seen_paths.add(path)
        seen_hashes.add(digest)
        splits[split].append(Document(item["path"], split, raw.decode("utf-8"), digest))
    if any(not documents for documents in splits.values()):
        raise ValueError("The manifest must contain nonempty train, validation, and test splits.")
    return splits


def split_fingerprint(splits: dict[str, list[Document]]) -> str:
    """Identify document identities, contents, and split membership across resumes."""
    records = sorted((doc.path, doc.split, doc.sha256)
                     for documents in splits.values() for doc in documents)
    return hashlib.sha256(json.dumps(records).encode("utf-8")).hexdigest()


class TokenWindows:
    """Turn separate encoded documents into fixed-width input/target windows.

    With stride n, each within-document next-token transition appears once.
    Tail windows use right padding: input ID 0, target IGNORE_INDEX. Causal
    attention prevents later padding from affecting valid earlier predictions.
    """

    def __init__(self, documents: Sequence[Tensor], n: int, *, stride: int | None = None):
        stride = n if stride is None else stride
        if n <= 0 or not 1 <= stride <= n:
            raise ValueError("n must be positive; stride must be between 1 and n.")
        self.n = n
        self.documents = list(documents)  # Separate 1D CPU integer tensors.
        self.windows: list[tuple[int, int]] = []  # (document index, first token index).
        for doc_idx, ids in enumerate(self.documents):
            if ids.ndim != 1 or ids.dtype != torch.long or ids.device.type != "cpu":
                raise ValueError("Encoded documents must be 1D CPU int64 tensors.")
            self.windows.extend((doc_idx, start) for start in range(0, len(ids)-1, stride))
        if not self.windows:
            raise ValueError("At least one document needs two tokens for next-token training.")

    def __len__(self) -> int:
        return len(self.windows)

    def batch(self, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
        if not indices:
            raise ValueError("A batch must contain at least one window.")
        inputs = torch.zeros(len(indices), self.n, dtype=torch.long)  # (B, n).
        targets = torch.full_like(inputs, IGNORE_INDEX)  # (B, n).
        for row, index in enumerate(indices):
            doc_idx, start = self.windows[index]
            span = self.documents[doc_idx][start:start+self.n+1]  # (up to n+1,).
            valid = len(span)-1  # Scalar number of supervised transitions.
            inputs[row, :valid] = span[:-1]
            targets[row, :valid] = span[1:]
        return inputs, targets

    def sample_batch(self, B: int, *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        if B <= 0:
            raise ValueError("B must be positive.")
        indices = torch.randint(len(self), (B,), generator=generator).tolist()  # B window indices.
        return self.batch(indices)

    def evaluation_batches(self, B: int, *, max_batches: int | None = None,
                           seed: int = 0) -> Iterator[tuple[Tensor, Tensor]]:
        """All windows, or a fixed reproducible subset; never consume training RNG."""
        if B <= 0 or (max_batches is not None and max_batches <= 0):
            raise ValueError("B and any max_batches must be positive.")
        indices = list(range(len(self)))
        if max_batches is not None and max_batches*B < len(self):
            rng = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(self), generator=rng)[:max_batches*B].tolist()
        for start in range(0, len(indices), B):
            yield self.batch(indices[start:start+B])


def make_batch(token_ids: Tensor, B: int, n: int,
               *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Convenience for ONE document; do not concatenate different documents here."""
    return TokenWindows([token_ids], n).sample_batch(B, generator=generator)
