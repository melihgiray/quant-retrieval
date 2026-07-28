# quant-retrieval

Semantic search over quantitative finance Q&A from quant.stackexchange.com.
Ask a question in plain English, get back the answers that actually address it.

The search box is the visible part. The work underneath is an embedding model
fine-tuned with a contrastive loss written in PyTorch, measured against BM25 and
off-the-shelf embeddings on held-out questions the model never saw.

Status: in progress. The data pipeline is first, then the evaluation harness and
baselines, then training. Results go in RESULTS.md as they are produced, and no
number appears there that did not come out of the evaluation harness.

## Data

Posts come from the public Stack Exchange data dump for quant.stackexchange.com,
published on archive.org. User contributions are licensed CC BY-SA 4.0 by their
authors. See docs/DATA.md for the dump date, the counts, and how questions are
split into train, validation, and test.
