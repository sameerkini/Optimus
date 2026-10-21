# Training corpora

Downloaded corpora, uploaded documents, prepared datasets, and generated checkpoints stay local and are excluded from version control. `smoke.txt` is a small original fixture for tokenization and fixed-batch checks; it is too small for a meaningful language-model evaluation.

## Optional Gutenberg corpus

The example corpus uses these Project Gutenberg editions:

| Book | Author | Source | Documents |
| --- | --- | --- | --- |
| Alice's Adventures in Wonderland | Lewis Carroll | [Gutenberg 11](https://www.gutenberg.org/ebooks/11) | 12 chapters |
| The Adventures of Sherlock Holmes | Arthur Conan Doyle | [Gutenberg 1661](https://www.gutenberg.org/ebooks/1661) | 12 stories |
| The Wonderful Wizard of Oz | L. Frank Baum | [Gutenberg 55](https://www.gutenberg.org/ebooks/55) | 24 chapters |

Download the full originals into `data/gutenberg/raw/`:

```sh
mkdir -p data/gutenberg/raw
curl --fail --location https://www.gutenberg.org/cache/epub/11/pg11.txt --output data/gutenberg/raw/pg11.txt
curl --fail --location https://www.gutenberg.org/cache/epub/1661/pg1661.txt --output data/gutenberg/raw/pg1661.txt
curl --fail --location https://www.gutenberg.org/cache/epub/55/pg55.txt --output data/gutenberg/raw/pg55.txt
python -m scripts.prepare_corpus
```

Run these commands from the project root. On macOS, `/usr/bin/curl` can be used if a separately installed curl has certificate-store problems. Keep original download files intact, including credits and license notices; consult each linked source before redistributing its text.

The preparation script assigns whole chapters/stories before tokenization, using a fixed seed of 42. The result is **40 training, four validation, and four test documents**. Each book contributes to all splits. This is held-out-section evaluation within familiar books, not evaluation on unseen books. Proportions are approximate by section count because lengths differ.

Derived documents exclude Gutenberg front matter and footer boilerplate while preserving chapter headings and interior text. CRLF becomes LF and outer whitespace is trimmed. `gutenberg/manifest.json` records source URLs, original/derived hashes, section identities, preparation rules, and split membership. The script verifies expected section counts and refuses to overwrite a changed derived document. Upstream editions can change; preserve the manifest and raw hashes when reproducing an existing experiment.

## Custom documents

Prepare distinct UTF-8 text documents and assign whole documents to training, validation, and test splits. Each split must contain at least one document; more substantial held-out material is needed for useful evaluation.

The CLI accepts a prepared corpus through `python -m src.train --manifest PATH`. A manifest contains a `documents` list; each entry has a corpus-relative `path`, `split` (`train`, `validation`/`val`, or `test`), and the file's SHA-256 as `sha256`. All splits must be nonempty. `src/data.py` rejects duplicate contents, duplicate/escaping paths, and changed files. Keep related or overlapping documents in the same split to avoid leakage beyond these exact-duplicate checks.

Fit the tokenizer only on training documents, then freeze its token-ID mapping for validation, test, checkpoints, and generation. Adding documents with a new vocabulary requires a new model run. The tokenizer counts text; language learning happens in the model's weight updates.

## Windows and statistics

Input spans contain up to `n+1` tokens. Inputs use all but the final token, while targets use all but the first. Windows never cross documents. With stride `n`, the complete window list covers each within-document next-token transition once. Training samples windows with replacement; evaluation walks the full list unless a fixed subset is requested. Short tails are padded with input ID zero and target `-100`; padded targets contribute no loss.

The tokenizer preserves case, punctuation, and whitespace:

- **Words:** occurrences of regex `\w+` runs, including digits and underscores; contractions may split.
- **Tokens:** every regex piece, including whitespace and punctuation.
- **Vocabulary `V`:** observed training token types plus reserved `<UNK>`.
- **Singleton types:** token types seen once, indicating vocabulary sparsity.
- **Next-token targets:** `sum(max(document_tokens - 1, 0))`, excluding document boundaries.
- **Unknown-token rate:** held-out tokens missing from the training vocabulary. Word-only rates help separate this from frequent whitespace.

`Tokenizer.show_stats(n=64)` reports corpus scale and window counts. Overlapping windows and repeated epochs reuse text; processed tokens are not fresh corpus tokens. A `V × d_emb` embedding table can dominate a small model's parameter count. Weight tying reuses its transpose for output scoring. Tokens-per-parameter ratios describe scale; use held-out learning curves to assess model capacity and data sufficiency.

Tokenizer vocabularies contain strings derived from the corpus. Keep documents and model exports private when the training text is private.
