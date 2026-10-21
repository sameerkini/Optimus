"""Split text into words, punctuation, and whitespace tokens."""

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


# Match every character: whitespace runs, word runs, or one punctuation/symbol.
_TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]")
_WORD_PATTERN = re.compile(r"\w+")


class Tokenizer:
    """A reversible tokenizer for known tokens; fit only on training text.

    Embedding vectors belong to the model, which learns them during training.
    This class stores token IDs and training counts only.
    """

    def __init__(self, training_text: str | Iterable[str]):
        """Fit one shared vocabulary on a string or an iterable of documents.

        Process documents separately so their boundaries do not create tokens
        or next-token targets. Retain counts, not the original documents.
        """
        self.unk_token = "<UNK>"
        self.unk_id = 0
        self.token_counts: Counter[str] = Counter()
        self.document_token_counts: list[int] = []
        self.total_characters = 0

        documents = [training_text] if isinstance(training_text, str) else training_text
        for document in documents:
            if not isinstance(document, str):
                raise TypeError("Each document must be text. Use from_files() for paths.")
            tokens = self.tokenize(document)
            self.token_counts.update(tokens)
            self.document_token_counts.append(len(tokens))
            self.total_characters += len(document)

        # Sorting makes IDs reproducible regardless of token encounter order.
        self.token_to_id = {self.unk_token: self.unk_id}
        for token_id, token in enumerate(sorted(self.token_counts), start=1):
            self.token_to_id[token] = token_id

        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "Tokenizer":
        """Read UTF-8 documents one at a time into one shared vocabulary.

        Pass training paths only. Encoding held-out text never refits this map.
        """
        return cls(Path(path).read_text(encoding="utf-8") for path in paths)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Preserve case, punctuation, symbols, and exact whitespace."""
        return _TOKEN_PATTERN.findall(text)

    @property
    def V(self) -> int:
        """Include the reserved unknown token in the model's vocabulary size."""
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """Map tokens to IDs, using the unknown ID for unseen tokens.

        Encoding never adds vocabulary entries or changes training counts.
        """
        return [
            self.token_to_id.get(token, self.unk_id)
            for token in self.tokenize(text)
        ]

    def decode(self, ids: list[int]) -> str:
        """Join tokens exactly; unseen encoded tokens decode as <UNK>.

        An invalid ID raises KeyError because it has no vocabulary entry.
        """
        return "".join(self.id_to_token[token_id] for token_id in ids)

    def state_dict(self) -> dict:
        """JSON-safe vocabulary/counts for checkpoints; IDs retain their exact order."""
        return {"tokens": [self.id_to_token[i] for i in range(self.V)],
                "token_counts": dict(self.token_counts),
                "document_token_counts": self.document_token_counts,
                "total_characters": self.total_characters}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Tokenizer":
        tokenizer = cls([])  # Create bookkeeping without fitting any new text.
        tokens = state["tokens"]
        if not tokens or tokens[0] != tokenizer.unk_token or len(set(tokens)) != len(tokens):
            raise ValueError("Saved vocabulary must be unique and start with <UNK>.")
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("Saved vocabulary entries must be strings.")
        tokenizer.token_to_id = {token: i for i, token in enumerate(tokens)}
        tokenizer.id_to_token = dict(enumerate(tokens))
        tokenizer.token_counts = Counter(state["token_counts"])
        tokenizer.document_token_counts = list(state["document_token_counts"])
        tokenizer.total_characters = state["total_characters"]
        return tokenizer

    def show_stats(
        self,
        n: int | None = None,
        top_k: int = 10,
        parameter_count: int | None = None,
    ) -> dict:
        """Print corpus statistics and return the same information as a dict.

        A word is a regex word run (letters, digits, or underscores). Model
        tokens also include punctuation/symbols and whitespace runs; all are
        valid token IDs. Each document contributes its token count minus one next-token
        targets (or zero for an empty document). The optional n argument is
        the input-window length for counting full-context examples. No boundary tokens or padding are added here.
        """
        if n is not None and n <= 0:
            raise ValueError("n must be positive.")
        if top_k < 0:
            raise ValueError("top_k cannot be negative.")
        if parameter_count is not None and parameter_count <= 0:
            raise ValueError("parameter_count must be positive.")

        total_tokens = self.token_counts.total()
        word_counts = {
            token: count for token, count in self.token_counts.items()
            if _WORD_PATTERN.fullmatch(token)
        }
        whitespace_tokens = sum(
            count for token, count in self.token_counts.items() if token.isspace()
        )
        total_words = sum(word_counts.values())
        lengths = self.document_token_counts
        stats = {
            "documents": len(lengths),
            "empty_documents": lengths.count(0),
            "total_characters": self.total_characters,
            "total_words": total_words,
            "unique_word_types": len(word_counts),
            "total_tokens": total_tokens,
            "whitespace_tokens": whitespace_tokens,
            "punctuation_symbol_tokens": total_tokens - total_words - whitespace_tokens,
            "non_whitespace_tokens": total_tokens - whitespace_tokens,
            "V": self.V,
            "observed_token_types": len(self.token_counts),
            "singleton_token_types": sum(count == 1 for count in self.token_counts.values()),
            "next_token_targets": sum(max(length - 1, 0) for length in lengths),
            "min_document_tokens": min(lengths, default=0),
            "max_document_tokens": max(lengths, default=0),
            "mean_document_tokens": total_tokens / len(lengths) if lengths else 0.0,
            "most_common_tokens": self.token_counts.most_common(top_k),
            "most_common_words": Counter(word_counts).most_common(top_k),
        }

        print("TRAINING CORPUS STATISTICS")
        print(f"Documents: {stats['documents']:,} ({stats['empty_documents']:,} empty)")
        print(f"Characters: {self.total_characters:,}")
        print(f"Words (regex word runs): {total_words:,}")
        print(f"Unique word types (case-sensitive): {len(word_counts):,}")
        print(f"Valid model tokens, including whitespace: {total_tokens:,}")
        print(f"  Whitespace tokens: {whitespace_tokens:,}")
        print(f"  Punctuation/symbol tokens: {stats['punctuation_symbol_tokens']:,}")
        print(f"  Non-whitespace tokens: {stats['non_whitespace_tokens']:,}")
        print(f"Vocabulary size, including <UNK>: {self.V:,}")
        print(f"Token types seen exactly once: {stats['singleton_token_types']:,}")
        print(f"Distinct within-document next-token targets: {stats['next_token_targets']:,}")
        print(
            "Document lengths in tokens (min / mean / max): "
            f"{stats['min_document_tokens']:,} / "
            f"{stats['mean_document_tokens']:,.1f} / {stats['max_document_tokens']:,}"
        )

        if n is not None:
            stats["n"] = n
            stats["full_context_sliding_windows"] = sum(
                max(length - n, 0) for length in lengths
            )
            stats["documents_too_short_for_context"] = sum(
                length <= n for length in lengths
            )
            print(
                f"Full-context sliding windows at n={n} (stride 1): "
                f"{stats['full_context_sliding_windows']:,}"
            )
            print("  Overlapping windows reuse data; this is not a unique-token count.")
            print(f"  Documents too short: {stats['documents_too_short_for_context']:,}")

        if parameter_count is not None:
            stats["parameter_count"] = parameter_count
            stats["corpus_tokens_per_parameter"] = total_tokens / parameter_count
            print(f"Parameters supplied: {parameter_count:,}")
            print(f"Corpus tokens per parameter: {stats['corpus_tokens_per_parameter']:.3f}")
            print("  A size diagnostic, not a guarantee of sufficient data or model quality.")

        print("Most common tokens (repr makes whitespace visible):")
        for token, count in stats["most_common_tokens"]:
            print(f"  {token!r}: {count:,}")
        print("Most common words:")
        for token, count in stats["most_common_words"]:
            print(f"  {token!r}: {count:,}")
        return stats
