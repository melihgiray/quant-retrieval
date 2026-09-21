# Comparing evaluation runs

Compare existing validation results from the repository root:

```sh
python -m scripts.compare_runs \
  --baseline results/minilm_frozen_val.json \
  --candidate results/minilm_tuned_epoch3_val.json \
  --metric ndcg_at_10 --iterations 10000 --seed 17 --confidence 0.95
```

Both files need per-query scores, the same named split, and matching query
counts, corpus sizes and retrieval cutoffs. Query IDs must match exactly. These
checks catch incompatible experiment settings; matching counts alone cannot
prove that two corpora contain identical documents.

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
rejected before comparison.
