"""Is a difference between two runs larger than the noise.

Every comparison in this project has been a bare subtraction. Hard negatives
bought +0.0008 nDCG@10 and CLS pooling cost -0.0738, and the write-ups call the
first nothing and the second real. Both readings are almost certainly right, and
until now nothing in the repo demonstrated either.

The test is a paired bootstrap. Resample the 753 validation questions with
replacement ten thousand times, and each time recompute both systems' averages
over the same resampled questions. The spread of the differences is the sampling
distribution, and if it comfortably excludes zero the gap is not an accident of
which questions happen to be in the set.

Paired matters. Both systems answered the same questions, so resampling questions
rather than scores keeps them aligned, and the large variance from some questions
simply being harder than others cancels instead of drowning the effect. Comparing
two unpaired samples of 753 numbers would need a far bigger gap to reach the same
confidence, and would be answering a question nobody asked.

This is a bootstrap, not a t-test, because these metrics are nothing like normal:
reciprocal rank lives on 1, 1/2, 1/3 and zero, and recall at 10 over a single
relevant document is a coin flip. Resampling assumes none of that.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULT_ITERATIONS = 10_000


def paired_bootstrap(
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict[str, float]:
    """Compare two systems question by question.

    Both mappings are query id to that query's score. Returns the observed mean
    difference (candidate minus baseline), a confidence interval for it, and a
    two sided p value for the difference being zero.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("the two runs share no queries")
    if len(shared) != len(baseline) or len(shared) != len(candidate):
        raise ValueError(
            f"the two runs cover different queries: {len(baseline)} and "
            f"{len(candidate)}, sharing {len(shared)}"
        )

    differences = np.array(
        [candidate[query_id] - baseline[query_id] for query_id in shared], dtype=np.float64
    )
    observed = float(differences.mean())

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(differences), size=(iterations, len(differences)))
    resampled = differences[draws].mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])

    # Centre the resampled differences on zero to get the distribution under the
    # hypothesis that the systems are identical, then ask how often that null
    # produces a difference at least as large as the one actually measured.
    centred = resampled - resampled.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "queries": len(shared),
        "mean_difference": observed,
        "confidence": confidence,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
        "iterations": iterations,
        "significant": bool(low > 0 or high < 0),
    }


def format_difference(result: Mapping[str, float]) -> str:
    """One line a results table can carry."""
    return (
        f"{result['mean_difference']:+.4f} "
        f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], "
        f"p = {result['p_value']:.3f}"
    )
