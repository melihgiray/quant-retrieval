# Auditing saved ANN measurements

The sweep measures index lookup over fixed vectors. It does not measure text
encoding, end-to-end search, relevance quality or a deployed service. The test
suite exercises tiny fixtures. There is no new real scaling result from those
tests, and a faster fixture search does not establish a production crossover.

## Run a sweep

Prepare exports with the same checkpoint, sequence length and pooling. A live
sweep on Linux can use the command in [benchmarking](BENCHMARKING.md). On macOS,
run encoding and search in separate processes using a portable query cache:

```sh
python -m scripts.export_queries --index artifacts/quant-v2 \
  --checkpoint checkpoints/minilm_tuned/epoch-3 --queries 200 --seed 17 \
  --output artifacts/queries-v1

python -m scripts.ann_sweep --embeddings artifacts/quant-v2 artifacts/scale_100000 \
  --query-cache artifacts/queries-v1 --k 10 --warmup 10 --repeats 3 --threads 1 \
  --ef-search 16 32 64 128 256 --output results/ann_scaling_new.json
```

These are two separate command invocations, not two calls in one notebook
kernel. The first loads the encoder but never FAISS. The second loads FAISS but
never the encoder, and needs neither model files nor the original query table.
All document exports must include production-time checkpoint hashes and array
checksums. The existing legacy export cannot be upgraded by adding current
hashes; first generate a new document export as described in
[index exports](INDEXES.md). No such real export is implied by the code tests.

Cache mode rejects `--data`, `--checkpoint`, `--queries` and `--seed` instead of
silently ignoring them. Sampling and encoder settings come from the cache.
Each document export must match its checkpoint hashes, pooling, maximum length
and dimensions. Different corpus sizes can reuse the same queries when those
settings match.

The macOS guard refuses live encoding inside the sweep or a cached sweep in a
process that has already loaded torch. It also refuses query encoding in a
process that already loaded FAISS. Start a fresh terminal command when this
happens. The incompatible runtimes themselves have not been repaired; the new
workflow avoids loading them together. A real FAISS subprocess test verifies
exact and HNSW retrieval over three synthetic documents on the local Mac.

Use a fresh output path. A previous report is preserved unless `--overwrite` is
explicit, and that option cannot target the index inputs or checkpoint files.
Repeated artifact paths and repeated search settings are rejected. Partial
reports are saved after each completed setting with `complete: false`; they are
not resumable or eligible for operating-point analysis.

Version 1 reports contain:

- The query IDs, query-text fingerprint, source revision and machine context.
- Index-only scope, thread count, graph settings, warmup and repetition counts.
- One exact row and one row per requested HNSW setting for each artifact.
- Individual millisecond timings ordered by repetition, then query.
- Recall per query, measured as overlap with the exact top-k on that corpus.
- Unrounded build times. HNSW settings share one graph build, not separate builds.
- Query source metadata and FAISS/NumPy versions. Cached encoding provenance
  stays under `query_source.manifest`, separate from the benchmark machine.

Warmup does not enter the saved timing samples. Repetition increases the number
of timings, not the number of independent questions. Exact top-k ties can admit
multiple equally scoring answer sets; overlap refers to the returned exact set,
not every mathematically equivalent ranking.

## Analyze without a model

```sh
python -m scripts.analyze_ann --report results/ann_scaling_new.json \
  --recall-target 0.95 --worst-queries 10 --output results/ann_analysis_new.json
```

This command loads neither the encoder runtime nor FAISS. It requires a complete
version 1 report, checks declared sweep coverage, and recomputes the percentiles
and mean recall from raw samples. Duplicate, missing or inconsistent rows fail
before output is published. Existing analysis files and the input measurement
file cannot be replaced. Legacy aggregate-only reports need a fresh sweep; do
not invent raw timings to make them pass the audit.

The analysis retains a SHA-256 of the source report and its measurement context.
For cached sweeps, it also checks query IDs, seed and fingerprint against the
cache metadata and checks encoding compatibility against each document export.
Older version 1 reports without `query_source` remain readable with their
earlier provenance limits.
Changing `--recall-target` analyzes the same measurements under another threshold;
it does not measure a new run or improve the model.
The optional `--max-p95-ms` similarly filters operating points against a chosen
nonnegative index-search tail-latency budget. It does not include query encoding,
network time, queueing, or response formatting, and is not an end-to-end service
guarantee. The selected budget is recorded in the analysis JSON.

## Read the output

`best_eligible_hnsw` is the lowest observed p50 among settings that meet the
recall target and optional p95 budget. It is null when none qualify. Ties use
p95, then search breadth. `observed_winner` selects the lowest p50 among eligible
options, preferring exact on a tie. It is null if neither option meets the budget.
`exact_meets_latency_budget` states whether the exact baseline is eligible.
`speedup` is exact p50 divided by the qualifying graph's p50, and is null when
the denominator is zero or no graph qualifies.

`estimated_queries_to_amortize_build` divides the graph's extra build time by
its observed p50 saving per query, converts seconds to milliseconds, and rounds
up. It is zero if a faster graph costs no more to build, and null when no graph
qualifies or there is no positive p50 saving. This is a rough serial-work estimate,
not a measured workload crossover: medians do not predict total request time,
and the calculation ignores concurrency, cache effects, rebuild frequency, and
memory cost. It can remain null for numeric values too large to represent.

The `pareto_frontier` retains points for which no measured alternative is both
at least as fast and at least as accurate, with one strict improvement. This
frontier considers p50 and exact-set recall only; it does not optimize build
time, memory or p95. Every comparison stays within one artifact. Equal document
counts do not make different corpora interchangeable.

Query diagnostics show the lowest-recall questions at every graph setting,
breaking ties by slower query p50 and then question ID. Per-query medians group
samples across repetitions in the declared order. Use the IDs to inspect the
original validation questions; these examples are not significance tests.

An observed p50 win is not proof of a stable latency advantage. Repeat studies
on a quiet machine and retain hardware, thread settings and raw reports. Claim
a corpus-size crossover only after measuring comparable nested corpora; the
analyzer neither infers unseen sizes nor certifies corpus nesting.
