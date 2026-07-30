# Results

These numbers come from the committed JSON files in `results/`. The evaluation ranks
all 26,152 answers for each query. Validation is used until the final model is fixed.
The test split has not been run.

## Validation baselines

| Retriever | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | Graded nDCG@10 | Index time (s) | Search p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4085 | 0.3692 | 0.5339 | 0.7384 | 0.4060 | 1.40 | 5.48 |
| Frozen MiniLM | 0.4962 | 0.4518 | 0.6375 | 0.8539 | 0.5045 | 45.82 | 5.05 |

Strict metrics count only the accepted answer. Graded nDCG also gives partial
credit to other nonnegative answers written for the same question.

BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is
`sentence-transformers/all-MiniLM-L6-v2`, used without domain training. It mean
pools nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by
cosine similarity.

Timing was measured on an Apple M5 Pro. Index time includes tokenization and
embedding for dense retrieval. Search latency includes query encoding and exact
scoring against the full corpus. These are baseline measurements, not the final
latency study.

## Reproduce

```sh
python scripts/evaluate.py --config configs/bm25.yaml
python scripts/evaluate.py --config configs/minilm_frozen.yaml
python scripts/make_results_table.py
```
