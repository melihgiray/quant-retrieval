# Auditing saved ANN measurements

The sweep measures index lookup over fixed vectors. It does not measure text
encoding, end-to-end search, relevance quality or a deployed service. The test
suite exercises tiny fixtures. There is no new real scaling result from those
tests, and a faster fixture search does not establish a production crossover.

## Run a sweep

Prepare exports with the same checkpoint, sequence length and pooling, then use
the command in [benchmarking](BENCHMARKING.md). The live sweep still needs a
machine where the encoder and FAISS runtimes coexist. The macOS runtime conflict
has not been removed by adding an offline analyzer.

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
Changing `--recall-target` analyzes the same measurements under another threshold;
it does not measure a new run or improve the model.

## Read the output

`best_eligible_hnsw` is the lowest observed p50 among settings that meet the
recall target. It is null when none qualify. Ties use p95, then search breadth.
`observed_winner` stays exact when the best qualifying graph is no faster.
`speedup` is exact p50 divided by the qualifying graph's p50, and is null when
the denominator is zero or no graph qualifies.

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
