# Audit and group saved evaluation evidence

These commands read existing train or validation reports. They do not load a
model, rerank answers, score the held-out split, or establish statistical
significance. Choose report files that contain all five per-query metrics.
Aggregate-only historical reports cannot supply missing query-level evidence.

## Check one report

```sh
python -m scripts.audit_run --run results/bm25_val_new.json \
  --output results/audits/bm25_val_new.json
```

The audit checks canonical query IDs, finite scores, recall across cutoffs,
strict top-ten hit consistency, query counts, and a retrieval depth sufficient
for Recall@100. It recomputes macro averages from the saved scores. The absolute
comparison tolerance is `1e-6`, to accommodate the writer's six-decimal query
scores. That tolerance describes serialization, not statistical uncertainty.

When present, diagnostics and rankings must cover the same questions. Counts,
first-primary ranks, reciprocal rank and Recall@100 are cross-checked where the
saved evidence permits it. Rankings need unique positive int64 IDs and cannot
exceed the recorded depth or corpus size. Empty rankings cannot support positive
scores. Timing summaries need finite nonnegative values and p95 at least p50.

Missing optional evidence is reported as absent. The audit validates the format
of dataset hashes and revision metadata but does not reopen original data or
verify a historical checkout. It cannot establish corpus membership of ranked
IDs or recompute relevance scores without the original labels. It does not
reconstruct latency percentiles from aggregate timing summaries.

## Inspect query groups

```sh
python -m scripts.slice_run --run results/bm25_val_new.json \
  --group-by first_primary_rank --output results/slices/bm25_ranks.json
```

Diagnostic groupings are `diagnostic_status`, `primary_labels` (zero, one, or at
least two primary labels), and `first_primary_rank` (first, second through tenth,
below tenth, or no hit). They require saved diagnostics. Status and rank groups
are defined by the observed outcome, so their averages are descriptive rather
than independent predictors of performance.

For text length, supply the exact query table used by the evaluation:

```sh
python -m scripts.slice_run --run results/bm25_val_new.json \
  --group-by query_length --queries data/processed/queries.parquet \
  --word-boundaries 10 25 50 100 --output results/slices/bm25_lengths.json
```

Length means whitespace-separated words, not model tokens. Boundaries are lower
bounds for the next bucket: `10 25` gives `0_to_9`, `10_to_24`, and `25_plus`.
The selected split must match the evaluation's exact query IDs, text and row
order fingerprint. Legacy runs without dataset hashes cannot establish this.
The report hashes the same captured parquet bytes used to assign the groups.

Every question belongs to one group, and each group records its IDs, count and
mean metrics. Groups smaller than `--min-queries` (default 5) remain visible with
`small_group: true`; they are not discarded or declared statistically reliable.

## Compare fixed groups

```sh
python -m scripts.slice_run --run results/bm25_val_new.json \
  --candidate results/hybrid_val_new.json --group-by diagnostic_status \
  --overlap-k 10 --output results/slices/bm25_vs_hybrid.json
```

Both reports must pass the audit and share the split, counts, query IDs and
dataset identities. Two legacy reports can be paired with
`dataset_identity_matched: false`. A fingerprinted report cannot be mixed with
a legacy one. Group assignments come from the baseline and stay fixed for both
systems. Per-group deltas use candidate minus baseline; query-change counts and
bounded examples show which saved scores improved, regressed, or stayed equal.

Outcome transitions account for every question when both reports contain
diagnostics. Ranking overlap is available when both saved rankings: it reports
Jaccard set overlap, baseline-answer retention and exact order agreement within
the selected cutoff. These are agreement measures, not relevance metrics. Pairs
of empty rankings have undefined Jaccard and are excluded from its mean, with
their count retained. The actual number of returned answers is always shown.

## Preserve evidence

Audit and slice outputs require a new path and cannot replace any input.
Outputs are published only after validation. Reports retain source paths and
SHA-256 identities; they do not copy query text. The test suite exercises the
commands on a tiny evaluated corpus and checks that fresh analysis processes
import neither the encoder nor the transformer runtime. Passing those fixtures
does not establish a new production quality result.
