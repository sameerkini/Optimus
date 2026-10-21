"""Print tokenizer behavior on real documents. No model or training required.

Run from the project folder: python -m scripts.inspect_tokenizer
"""

import argparse
import json
import re
from pathlib import Path

from src.tokenizer import Tokenizer


CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "gutenberg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--parameters", type=int, default=None,
                        help="Optional actual model parameter count for a size diagnostic.")
    parser.add_argument("--preview-tokens", type=int, default=30)
    args = parser.parse_args()
    if args.preview_tokens <= 0:
        parser.error("--preview-tokens must be positive")

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    train_paths = sorted(CORPUS_DIR / entry["path"] for entry in manifest["documents"]
                         if entry["split"] == "train")
    print(f"Fitting one vocabulary across {len(train_paths)} training documents.")
    tokenizer = Tokenizer.from_files(train_paths)
    stats = tokenizer.show_stats(n=args.n,
                                 parameter_count=args.parameters)

    print("\nFIRST TRAINING DOCUMENT:", train_paths[0].name)
    text = train_paths[0].read_text(encoding="utf-8")
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print("Source preview:", repr(text[:180]))
    print("Tokens, IDs, and training counts:")
    for token, token_id in zip(tokens[:args.preview_tokens], ids[:args.preview_tokens]):
        print(f"  {token!r:24} -> ID {token_id:5} | corpus count {tokenizer.token_counts[token]:,}")
    print("Decoded token preview:", repr(tokenizer.decode(ids[:args.preview_tokens])))
    print("Exact full-document round trip:", tokenizer.decode(ids) == text)
    for path in train_paths:
        document = path.read_text(encoding="utf-8")
        assert tokenizer.decode(tokenizer.encode(document)) == document, path.name
    print("Exact round trips for ALL training documents:", True)

    print("\nHELD-OUT VOCABULARY COVERAGE (training vocabulary stays fixed)")
    for split in ("validation", "test"):
        total = unknown = word_total = unknown_words = 0
        examples = set()
        split_documents = [entry for entry in manifest["documents"] if entry["split"] == split]
        for entry in split_documents:
            held_out_text = (CORPUS_DIR / entry["path"]).read_text(encoding="utf-8")
            for token in tokenizer.tokenize(held_out_text):
                is_unknown = token not in tokenizer.token_to_id
                total += 1
                unknown += is_unknown
                if re.fullmatch(r"\w+", token):
                    word_total += 1
                    unknown_words += is_unknown
                if is_unknown:
                    examples.add(token)
        print(f"{split}: {len(split_documents)} documents, {total:,} tokens; "
              f"<UNK> = {unknown:,}/{total:,} ({unknown / total:.2%})")
        print(f"  Unknown word runs: {unknown_words:,}/{word_total:,} ({unknown_words / word_total:.2%})")
        print("  Unknown token examples:", sorted(examples)[:12])
    assert tokenizer.V == stats["V"]
    print("Vocabulary size after held-out inspection:", tokenizer.V)
    print("\nOnly fitting/tokenization/statistics ran. Training can be inspected separately with python -m src.train --dry-run.")


if __name__ == "__main__":
    main()
