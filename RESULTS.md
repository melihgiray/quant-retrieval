# Results

These numbers come from the committed JSON files in `results/`. The evaluation ranks
all 26,152 answers for each query. Validation is used until the final model is fixed.
The test split has not been run.

## Validation results

| Retriever | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | Graded nDCG@10 | Index time (s) | Search p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4085 | 0.3692 | 0.5339 | 0.7384 | 0.4060 | 1.40 | 5.48 |
| Frozen MiniLM | 0.4962 | 0.4518 | 0.6375 | 0.8539 | 0.5045 | 45.82 | 5.05 |
| Tuned MiniLM (epoch 1) | 0.5210 | 0.4760 | 0.6627 | 0.8884 | 0.5280 | 39.69 | 4.81 |
| Tuned MiniLM (epoch 2) | 0.5343 | 0.4899 | 0.6746 | 0.8977 | 0.5411 | 39.65 | 4.85 |
| Tuned MiniLM (epoch 3) | 0.5358 | 0.4915 | 0.6760 | 0.8924 | 0.5427 | 39.40 | 4.71 |

Fine-tuning moves nDCG@10 from 0.4962 to 0.5358, +0.0396
absolute and +8.0% relative against the same encoder
untrained. Recall@100 is the number to watch for the reranking stage later, since
nothing a reranker does can recover an answer that never made the candidate list.

Strict metrics count only the accepted answer. Graded nDCG also gives partial
credit to other nonnegative answers written for the same question.

## What each row is

BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is
`sentence-transformers/all-MiniLM-L6-v2`, used without domain training. It mean
pools nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by
cosine similarity.

The tuned rows are that same encoder fine-tuned on the 9,924 training pairs with
an in-batch contrastive loss: each question is pulled toward its own answer and
pushed away from the other 63 answers in its batch. Three epochs, batch size 64,
AdamW at 2e-5 with linear warmup over the first tenth of steps and linear decay
after, gradient clipping at 1.0, temperature 0.05. Training took about eight
minutes on an Apple M5 Pro.

All three epoch checkpoints are listed because the checkpoint was chosen on
validation, and showing the choice is more useful than asserting it. The gain
flattens between epoch 2 and epoch 3, and epoch 2 is actually the better
checkpoint on Recall@100, so the ranking depends on which metric is being served.

Timing was measured on an Apple M5 Pro. Index time includes tokenization and
embedding for dense retrieval. Search latency includes query encoding and exact
scoring against the full corpus. These are not the final latency study.

## Reproduce

```sh
python scripts/evaluate.py --config configs/bm25.yaml
python scripts/evaluate.py --config configs/minilm_frozen.yaml
python scripts/train.py --config configs/base.yaml
python scripts/evaluate.py --config configs/minilm_tuned_epoch3.yaml
python scripts/make_results_table.py
```
