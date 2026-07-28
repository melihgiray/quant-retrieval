# quant-retrieval

Semantic search over quantitative finance Q&A from quant.stackexchange.com.
Ask a question in plain English, get back the answers that actually address it.

The search box is the visible part. The work underneath is an embedding model
fine-tuned with a contrastive loss written in PyTorch, measured against BM25 and
off-the-shelf embeddings on held-out questions the model never saw.

Status: in progress. The data pipeline is first, then the evaluation harness and
baselines, then training. Results go in RESULTS.md as they are produced, and no
number appears there that did not come out of the evaluation harness.

## Running it

Python 3.11 or newer.

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    python scripts/download_data.py
    python scripts/build_dataset.py

The first script pulls the 55 MB dump from archive.org. The second turns it into
`data/processed/{corpus,queries,qrels}.parquet` and writes the counts to
`results/dataset_stats.json`. Together they take about a minute.

    pytest

## Data

Posts come from the public Stack Exchange data dump for quant.stackexchange.com,
published on archive.org. The dump in use is dated 6 April 2024. User
contributions are licensed CC BY-SA 4.0 by their authors, and the dump itself is
not redistributed here.

The corpus is 26,152 answers. Queries are 11,474 questions, split by date so the
test set is the most recent questions rather than a random sample. docs/DATA.md
explains how answers are judged, how the splits avoid leaking duplicate
questions across the boundary, and what the known limitations are.
