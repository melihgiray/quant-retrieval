"""Test whether the gap between two runs is bigger than the noise.

    python scripts/compare_runs.py \\
        --baseline results/minilm_frozen_val.json \\
        --candidate results/minilm_tuned_epoch3_val.json

Writes results/comparisons/<baseline>_vs_<candidate>_<metric>.json so the
generated results table can quote an interval instead of a bare subtraction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from quant_retrieval.eval.metrics import METRIC_NAMES
from quant_retrieval.eval.significance import format_difference, paired_bootstrap


def load_per_query(path: Path, metric: str) -> dict[int, float]:
    result = json.loads(path.read_text())
    per_query = result.get("per_query") if isinstance(result, dict) else None
    if not isinstance(per_query, dict) or not per_query:
        raise SystemExit(
            f"{path} has no per_query block. It was written before per-query scores "
            "were kept, so rerun that evaluation before comparing it."
        )
    loaded = {}
    for query_id, scores in per_query.items():
        if not query_id.isascii() or not query_id.isdecimal() or str(int(query_id)) != query_id:
            raise SystemExit(f"{path} has an invalid query ID: {query_id!r}")
        if int(query_id) < 1:
            raise SystemExit(f"{path} has an invalid query ID: {query_id!r}")
        if not isinstance(scores, dict) or metric not in scores:
            raise SystemExit(f"{path} has no {metric!r} for query {query_id}")
        value = scores[metric]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise SystemExit(f"{path} has an invalid {metric!r} score for query {query_id}")
        loaded[int(query_id)] = value
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", default="ndcg_at_10", choices=METRIC_NAMES)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("results/comparisons"))
    args = parser.parse_args()

    baseline = load_per_query(args.baseline, args.metric)
    candidate = load_per_query(args.candidate, args.metric)
    result = paired_bootstrap(
        baseline, candidate, iterations=args.iterations, seed=args.seed
    )

    record = {
        "baseline": args.baseline.stem,
        "candidate": args.candidate.stem,
        "metric": args.metric,
        "baseline_mean": sum(baseline.values()) / len(baseline),
        "candidate_mean": sum(candidate.values()) / len(candidate),
        **result,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"{args.baseline.stem}_vs_{args.candidate.stem}_{args.metric}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output}")
    print(
        f"{args.metric}: {record['baseline_mean']:.4f} -> {record['candidate_mean']:.4f}, "
        f"{format_difference(result)}"
    )
    print("significant" if result["significant"] else "NOT significant at this level")


if __name__ == "__main__":
    main()
