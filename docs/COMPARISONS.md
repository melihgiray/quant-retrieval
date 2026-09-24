# Comparing evaluation runs

Compare existing validation results from the repository root:

```sh
python -m scripts.compare_runs \
  --baseline results/minilm_frozen_val.json \
  --candidate results/minilm_tuned_epoch3_val.json \
  --metric ndcg_at_10 --iterations 10000 --seed 17 --confidence 0.95
```

Both files need per-query scores, the same named split, and matching query
counts, corpus sizes and retrieval cutoffs. Query IDs must match exactly.

New evaluation records also contain `dataset_sha256` fingerprints for the corpus
IDs and text, selected query IDs and text, and selected relevance judgments.
Fingerprints preserve row order, including the index's document order, but ignore
DataFrame index labels. Changing a document, question, label or row order makes
the corresponding fingerprint change. Comparisons require all three identities
to match and set `dataset_identity_verified` to true in the report.

Two legacy records without fingerprints can still be compared; their report
sets `dataset_identity_verified` to false. Matching counts alone cannot establish
their data identity. Mixing a legacy record with a fingerprinted record is
rejected; rerun both evaluations on the same intended split to produce matching
provenance. Do not add fingerprints to old results based on today's data.

The paired bootstrap resamples the same questions for both systems and reports
the candidate's mean score minus the baseline's, a confidence interval and a
two-sided p value. The seed makes resampling reproducible. Samples are processed
in batches to limit memory use without changing the seeded results.

The report goes to `results/comparisons/` unless `--out` selects another directory.
It records the seed, iteration count, confidence level, split and SHA-256 hashes
of the exact input files. Keep those files with the report so later changes can
be detected. Existing reports remain intact if writing a replacement fails.

This command reads saved scores. It does not run a model or evaluate the test
split. Non-finite scores, ambiguous query IDs and scores outside [0, 1] are
rejected before comparison. Duplicate JSON keys are rejected rather than letting
the last occurrence silently replace an earlier score or metadata field.

New reports include `query_changes`: counts of improved, regressed and unchanged
questions, with the ten largest improvements and regressions. Ties mean exactly
equal saved scores. This is descriptive evidence for choosing examples to read,
not a separate significance test. See [error inspection](BENCHMARKING.md) for
ranking exports and the distinction between retrieval and ordering failures.
