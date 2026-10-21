"""Prepare chapter/story documents from three already-downloaded Gutenberg books.

This utility prepares data only. It does not implement batching or training.
Run from the project folder: python scripts/prepare_corpus.py
"""

import hashlib
import json
import random
import re
from pathlib import Path


CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "gutenberg"
SPLIT_SEED = 42
BOOKS = [
    {"id": 11, "name": "alice", "title": "Alice's Adventures in Wonderland",
     "author": "Lewis Carroll", "pattern": r"^CHAPTER [IVXLC]+\.$", "expected": 12},
    {"id": 1661, "name": "sherlock", "title": "The Adventures of Sherlock Holmes",
     "author": "Arthur Conan Doyle", "pattern": r"^[IVXLC]+\. [A-Z].+$", "expected": 12},
    {"id": 55, "name": "oz", "title": "The Wonderful Wizard of Oz",
     "author": "L. Frank Baum", "pattern": r"^Chapter [IVXLC]+$", "expected": 24},
]


def main() -> None:
    documents = []
    sources = []
    seen_hashes = set()
    for book in BOOKS:
        raw_path = CORPUS_DIR / "raw" / f"pg{book['id']}.txt"
        raw_bytes = raw_path.read_bytes()
        text = raw_bytes.decode("utf-8-sig").replace("\r\n", "\n")
        start = re.search(r"(?m)^\*\*\* START OF .+\*\*\*\s*$", text)
        end = re.search(r"(?m)^\*\*\* END OF .+\*\*\*\s*$", text)
        if start is None or end is None or start.end() >= end.start():
            raise ValueError(f"Missing Gutenberg boundary markers: {raw_path}")
        body = text[start.end():end.start()]
        headings = list(re.finditer(book["pattern"], body, flags=re.MULTILINE))
        if len(headings) != book["expected"]:
            raise ValueError(f"Expected {book['expected']} sections, found {len(headings)} in {raw_path}")

        # Assign whole sections, keeping each book represented in every split.
        section_ids = list(range(len(headings)))
        random.Random(SPLIT_SEED + book["id"]).shuffle(section_ids)
        held_out_count = max(1, round(len(headings) * 0.1))
        validation_ids = set(section_ids[:held_out_count])
        test_ids = set(section_ids[held_out_count:2 * held_out_count])

        for index, heading in enumerate(headings):
            stop = headings[index + 1].start() if index + 1 < len(headings) else len(body)
            document = body[heading.start():stop].strip() + "\n"
            digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                raise ValueError("Duplicate document found; inspect corpus before proceeding.")
            seen_hashes.add(digest)
            split = "validation" if index in validation_ids else "test" if index in test_ids else "train"
            relative_path = Path(split) / f"{book['name']}_{index + 1:02d}.txt"
            destination = CORPUS_DIR / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_text(encoding="utf-8") != document:
                raise ValueError(f"Refusing to overwrite changed corpus document: {destination}")
            destination.write_text(document, encoding="utf-8")
            documents.append({"path": relative_path.as_posix(), "split": split,
                              "book_id": book["id"], "section": index + 1,
                              "heading": heading.group(), "characters": len(document),
                              "sha256": digest})

        sources.append({"id": book["id"], "title": book["title"], "author": book["author"],
                        "page_url": f"https://www.gutenberg.org/ebooks/{book['id']}",
                        "download_url": f"https://www.gutenberg.org/cache/epub/{book['id']}/pg{book['id']}.txt",
                        "raw_path": raw_path.relative_to(CORPUS_DIR).as_posix(),
                        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "copyright_status": "Public domain in the USA (Project Gutenberg catalog)"})

    manifest = {"split_seed": SPLIT_SEED, "sources": sources, "documents": documents,
                "preparation": "Keep raw files intact. Extract sections after front matter and before Gutenberg footer; normalize CRLF to LF and outer whitespace only.",
                "split_policy": "Whole chapters/stories held out within each book, about 80/10/10 by section count. In-domain evaluation; not unseen-book evaluation."}
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in ("train", "validation", "test"):
        selected = [item for item in documents if item["split"] == split]
        print(f"{split}: {len(selected)} documents, {sum(item['characters'] for item in selected):,} characters")
    print(f"Source URLs, file hashes, and split assignments: {manifest_path}")


if __name__ == "__main__":
    main()
