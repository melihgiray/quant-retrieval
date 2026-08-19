# quant-retrieval

Semantic search over quantitative finance Q&A from quant.stackexchange.com.
Ask a question in plain English, get back the answers that actually address it.

The search box is the visible part. The work underneath is an embedding model
fine-tuned with a contrastive loss written in PyTorch, measured against BM25 and
off-the-shelf embeddings on held-out questions the model never saw.

Status: in progress. The dataset, the evaluation harness, both baselines, the
fine-tuned model, a round of ablations, hybrid retrieval, and a cross-encoder
reranker are all built and measured, and every comparison now carries a
confidence interval from a paired bootstrap. A hosted demo is not started.
Results land in RESULTS.md as they are produced, and no number appears there
that did not come out of the harness.

## Where it stands

On the validation split, ranking all 26,152 answers for each of 753 questions:

| | nDCG@10 | Recall@100 |
| --- | ---: | ---: |
| BM25 | 0.4085 | 0.7384 |
| MiniLM, off the shelf | 0.4962 | 0.8539 |
| MiniLM, fine-tuned here | 0.5358 | 0.8924 |
| BM25 and the tuned model, fused | 0.5550 | 0.9097 |

Fine-tuning is worth +0.0396 nDCG@10 over the same encoder untrained, after about
eight minutes of training on a laptop, and that gap survives a paired bootstrap
comfortably (p < 0.001).

Fusing that model with BM25 looks like another 4 percent, and the bootstrap says
otherwise: +0.0192 with an interval spanning zero. What fusion does buy, and what
does hold up, is Recall@100 rising to 0.9097 (p = 0.04). It finds answers the
encoder alone misses rather than ordering them better, which is the more useful
half anyway, since nothing downstream can rank a document that was never
retrieved. The test split has not been run.

RESULTS.md has the full table and what each row means. Four findings worth
skipping to, two of which are negative:

- Batch size is part of the objective, not just a speed knob, because in-batch
  negatives are the training signal. It climbs from 16 to 64, though 32 and 64
  are indistinguishable on this validation set.
- CLS pooling is much worse than mean pooling here, worse even than not training
  at all, because the base model was distilled with mean pooling.
- Mining hard negatives, the obvious next gain, did nothing for ranking and cost
  recall. Written up with why, rather than dropped.
- The cross-encoder reranker makes every pipeline worse, and undertraining is
  not the reason. Four measurements ruled that out and pointed at the training
  distribution instead: it had only ever seen mined hard negatives, so it never
  learned to reject an off-topic document. Retraining with random negatives
  mixed in took it from 43 to 85 percent against random distractors and
  improved every pipeline, and it still costs more than it gives. The diagnosis
  was right and the fix was not enough, which RESULTS.md says in those words.

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

Pipelines are described entirely by config. A reranking run nests its base
retriever, which can itself be a fusion of two others, and the whole tree is
written into the result file as provenance:

    python scripts/mine_negatives.py --config configs/negatives.yaml
    python scripts/train_reranker.py --config configs/reranker.yaml
    python scripts/evaluate.py --config configs/hybrid_rerank.yaml

Every experiment is a config file plus a seed, and reruns reproduce their
committed numbers exactly. `./run_ablations.sh` takes run names and works
through them in order.

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
