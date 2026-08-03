# quant-retrieval

Semantic search over quantitative finance Q&A from quant.stackexchange.com.
Ask a question in plain English, get back the answers that actually address it.

The search box is the visible part. The work underneath is an embedding model
fine-tuned with a contrastive loss written in PyTorch, measured against BM25 and
off-the-shelf embeddings on held-out questions the model never saw.

Status: in progress. The dataset, the evaluation harness, both baselines, and the
first fine-tuned model are done. A reranker, hybrid retrieval, an approximate
index, and a hosted demo are not. Results land in RESULTS.md as they are
produced, and no number appears there that did not come out of the harness.

## Where it stands

On the validation split, ranking all 26,152 answers for each of 753 questions:

| | nDCG@10 | Recall@100 |
| --- | ---: | ---: |
| BM25 | 0.4085 | 0.7384 |
| MiniLM, off the shelf | 0.4962 | 0.8539 |
| MiniLM, fine-tuned here | 0.5358 | 0.8924 |

Fine-tuning is worth 8 percent relative nDCG@10 over the same encoder untrained,
after about eight minutes of training on a laptop. RESULTS.md has the full table,
every epoch checkpoint, and what each row means. The test split has not been run.

## Running it

Python 3.11 or newer.

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    python scripts/download_data.py
    python scripts/build_dataset.py

The first script pulls the 55 MB dump from archive.org. The second turns it into
`data/processed/{corpus,queries,qrels}.parquet` and writes the counts to
`results/dataset_stats.json`. Together they take about a minute.

Then train and score a model:

    python scripts/train.py --config configs/base.yaml
    python scripts/evaluate.py --config configs/minilm_tuned_epoch3.yaml
    python scripts/make_results_table.py

Training writes one checkpoint per epoch and resumes with `--resume` if it stops.
`configs/smoke.yaml` runs the same loop on 512 pairs in about fifteen seconds,
which is the fast way to check a change before starting a real run.

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
