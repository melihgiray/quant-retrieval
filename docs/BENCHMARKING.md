# Performance studies and error inspection

These commands prepare and measure experiments. The checks in the test suite
use small fixtures; passing them is not a new model-quality or latency result.
Keep tuning on validation queries. Real model runs and large archive downloads
need enough memory, disk space and an otherwise quiet machine.

Evaluation and profiling reject duplicate YAML fields, including nested model
parameters. Keep each setting explicit once rather than relying on a later
duplicate to override an earlier value. See [index exports](INDEXES.md) for
atomic export publication, integrity checks and saved-precision evaluation.

## Profile a pipeline

```sh
python -m scripts.profile_pipeline --config configs/hybrid.yaml \
  --queries 100 --warmup 10 --repeats 3 --output results/hybrid_profile_new.json
```

Sampling uses the configured seed and is stable when input rows are reordered.
Only train and val splits are accepted. Warmup searches are excluded from stage
statistics. Repeats add timing observations, not new independent questions.
The report records question IDs, a digest of their text and order, configuration,
source revision, environment and individual query timings. Index construction is
reported separately. Parent stage times include children, so do not sum rows.
Shared retriever objects appear under their first path and are wrapped once.

Choose a new output filename to preserve an earlier profile. Report writes are
atomic, but profiling commands deliberately replace the specified report.

## Build nested distractor corpora

```sh
python -m scripts.build_scaling_corpus --sizes 100000 400000 --seed 17
python -m scripts.export_index --checkpoint checkpoints/minilm_tuned/epoch-3 \
  --corpus artifacts/scaling_corpus_100000.parquet --out artifacts/scale_100000
```

The original quant answers remain first and unchanged. Extra answers are sorted
by namespaced ID, shuffled once, then sampled by prefixes. A smaller corpus is
therefore a prefix of a larger one for the same sources and seed. Duplicate
answers, colliding site ID blocks and sizes outside available capacity fail
instead of silently changing the request. Only Posts.xml is extracted.

The summary records source file hashes, ID offsets, seed, revision, requested
sizes and output hashes. Each parquet file is replaced only after writing
finishes. A summary with `complete: false` describes partial work, not a finished
study. Old output files can remain in the directory; use the summary's file list
and hashes rather than assuming every file belongs to this run.

These extra sites are distractors, not new relevance judgments. The study
measures index approximation on fixed embeddings, not quality on those sites.

## Compare exact and approximate search

Install the optional runtime with `pip install -e ".[ann,dev]"`, then run on a
machine where FAISS and the encoder runtime coexist:

```sh
python -m scripts.ann_sweep --embeddings artifacts artifacts/scale_100000 \
  --checkpoint checkpoints/minilm_tuned/epoch-3 --queries 200 \
  --warmup 10 --repeats 3 --threads 1 --neighbours 32 --ef-construction 200 \
  --ef-search 16 32 64 128 256 --recall-target 0.95 \
  --output results/ann_scaling_new.json
```

Every artifact must agree on encoder length, dimensions and pooling. Manifest
counts must match unique positive IDs and normalized float32 vectors. New
exports verify model and tokenizer hashes as well as array checksums. Legacy
exports compare checkpoint paths instead and explicitly lack content proof.
Preserve the actual model and artifact files alongside a report.

The encoder runs once. Search latency includes query-vector normalization and
index lookup, but not text encoding. Both exact and approximate indexes use the
same vectors. One HNSW graph is built per corpus and reused across search-breadth
settings; its repeated build-time field is metadata, not another build.

Recall is overlap with exact top-k results, not labeled relevance. Timings retain
full precision in JSON. The summary selects the fastest eligible HNSW setting
within each artifact, never mixing two equal-sized corpora. Report hardware and
thread settings when quoting a result. Warm-cache repeats and a fixed sweep
order do not eliminate thermal drift or background activity.

Completed points are saved after each measurement. If the command fails, inspect
`complete` before using the report. This preserves partial evidence but does not
resume a sweep. Use a new filename for another run.

Existing sweep outputs need explicit `--overwrite`; input paths remain protected
even with that flag. Exact and approximate indexes are released between builds
to avoid keeping both full indexes in memory. See [offline ANN analysis](ANN.md)
for raw sample auditing, operating points and query-level diagnostics.

## Inspect retrieval errors

```sh
python -m scripts.evaluate --config configs/bm25.yaml --save-rankings \
  --output results/bm25_val_new.json
python -m scripts.inspect_errors --run results/bm25_val_new.json \
  --metric ndcg_at_10 --limit 20 --output results/bm25_val_errors.json
```

Evaluation refuses an existing output unless `--overwrite` is explicit. The
final test evaluation additionally requires `--allow-test`; do not use it while
tuning. The error-inspection command accepts train and validation records only.
It reads saved results without loading a model, and cannot replace its input.

New results distinguish a primary answer in the top ten, one below that cutoff,
one absent from the returned ranking, and a question without a primary label.
They count sibling and unjudged answers separately. Unjudged does not mean
irrelevant. Optional saved rankings contain ordered answer IDs for manual lookup.
Legacy records without diagnostics need a new validation evaluation; their
missing evidence cannot be reconstructed from aggregate scores.

Reranked heads use cross-encoder logits. Untouched tail items keep their base
ordering and receive the lowest head logit's score as an ordering sentinel.
Those tail scores are not model predictions and must not be interpreted as
confidence or compared with the base retriever's score scale.

Paired comparison reports also show counts of improved, regressed and unchanged
questions, plus the largest observed changes. These examples help investigate
failure modes. They are not significance tests on individual questions; the
paired bootstrap remains the aggregate uncertainty estimate.

When rankings are present, error reports include bounded `top_answer_ids` lists
for the displayed questions. `--rank-limit` controls this preview independently
of `--limit`, which controls the number of questions. Rank lists, retrieval
counts and diagnostic status must agree before a report is generated.

## Reranker sanity probes

```sh
python -m scripts.probe_reranker --checkpoint checkpoints/reranker_mixed/epoch-2 \
  --questions 100 --distractors 4 --seed 17 --output results/reranker_probe_new.json
```

This does run the reranker model, unlike offline error inspection. Questions
are sampled reproducibly from train and validation queries with one primary
answer. Distractors exclude all judged answers and all answers to the same
question. A pool too small for the request fails instead of admitting duplicates.

The positive must strictly outscore every distractor to count as a win. Ties are
reported separately; listing the positive first cannot award it a tied win.
The chance field is the continuous-score random-ranking reference, not an
expected strict-win rate for a constant scorer. Scores must be finite and have
one value per candidate. Reports retain sample identity, data fingerprints,
seed and settings, and require `--overwrite` to replace an existing result.

These rules differ from the earlier prefix-sampled probe protocol, which could
draw relevant answers and award positive-position ties. Historical probe numbers
have not been rerun under the corrected protocol. Do not compare them as though
only model weights changed, or rewrite earlier result files without a new run.
