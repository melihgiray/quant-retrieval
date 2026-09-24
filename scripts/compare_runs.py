"""Test whether the gap between two runs is bigger than the noise.

    python scripts/compare_runs.py \\
        --baseline results/minilm_frozen_val.json \\
        --candidate results/minilm_tuned_epoch3_val.json

Writes results/comparisons/<baseline>_vs_<candidate>_<metric>.json so the
generated results table can quote an interval instead of a bare subtraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from quant_retrieval.eval.analysis import paired_query_changes
from quant_retrieval.eval.metrics import METRIC_NAMES
from quant_retrieval.eval.results import write_result
from quant_retrieval.eval.significance import format_difference, paired_bootstrap


def parse_record(contents: bytes, path: Path) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        record = json.loads(contents, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise SystemExit(f"{path} is not a valid result record: {error}") from error
    if not isinstance(record, dict):
        raise SystemExit(f"{path} result record must be an object")
    return record


def load_per_query(path: Path, metric: str, *, record: dict | None = None) -> dict[int, float]:
    result = parse_record(path.read_bytes(), path) if record is None else record
    per_query = result.get("per_query") if isinstance(result, dict) else None
    if not isinstance(per_query, dict) or not per_query:
        raise SystemExit(
            f"{path} has no per_query block. It was written before per-query scores "
            "were kept, so rerun that evaluation before comparing it."
        )
    loaded = {}
    for query_id, scores in per_query.items():
        if (
            not isinstance(query_id, str) or not query_id.isascii()
            or not query_id.isdecimal() or str(int(query_id)) != query_id
        ):
            raise SystemExit(f"{path} has an invalid query ID: {query_id!r}")
        if int(query_id) < 1:
            raise SystemExit(f"{path} has an invalid query ID: {query_id!r}")
        if not isinstance(scores, dict) or metric not in scores:
            raise SystemExit(f"{path} has no {metric!r} for query {query_id}")
        value = scores[metric]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)
        ):
            raise SystemExit(f"{path} has an invalid {metric!r} score for query {query_id}")
        loaded[int(query_id)] = value
    return loaded


def validate_comparable_runs(baseline: dict, candidate: dict) -> None:
    for record in (baseline, candidate):
        if not isinstance(record.get("split"), str) or not record["split"].strip():
            raise SystemExit("comparison requires a named split string")
        if not isinstance(record.get("counts"), dict):
            raise SystemExit("comparison requires a counts object")
    if not baseline.get("split") or baseline.get("split") != candidate.get("split"):
        raise SystemExit("comparison requires runs from the same named split")
    for name in ("corpus_documents", "max_results", "queries"):
        left = baseline.get("counts", {}).get(name)
        right = candidate.get("counts", {}).get(name)
        if type(left) is not int or left < 1 or left != right:
            raise SystemExit(f"comparison requires matching positive {name} counts")
    left_hashes, right_hashes = baseline.get("dataset_sha256"), candidate.get("dataset_sha256")
    if left_hashes is None and right_hashes is None:
        return
    for hashes in (left_hashes, right_hashes):
        if (
            not isinstance(hashes, dict) or set(hashes) != {"corpus", "queries", "qrels"}
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in hashes.values())
        ):
            raise SystemExit("comparison requires complete dataset fingerprints in both runs")
    if left_hashes != right_hashes:
        raise SystemExit("comparison dataset fingerprints differ; rerun on identical data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", default="ndcg_at_10", choices=METRIC_NAMES)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=Path("results/comparisons"))
    args = parser.parse_args()

    baseline_bytes = args.baseline.read_bytes()
    candidate_bytes = args.candidate.read_bytes()
    baseline_record = parse_record(baseline_bytes, args.baseline)
    candidate_record = parse_record(candidate_bytes, args.candidate)
    baseline = load_per_query(args.baseline, args.metric, record=baseline_record)
    candidate = load_per_query(args.candidate, args.metric, record=candidate_record)
    validate_comparable_runs(baseline_record, candidate_record)
    if len(baseline) != baseline_record["counts"]["queries"]:
        raise SystemExit("baseline query count does not match per_query scores")
    if len(candidate) != candidate_record["counts"]["queries"]:
        raise SystemExit("candidate query count does not match per_query scores")
    result = paired_bootstrap(
        baseline, candidate, iterations=args.iterations, seed=args.seed, confidence=args.confidence
    )

    record = {
        "baseline": args.baseline.stem,
        "candidate": args.candidate.stem,
        "metric": args.metric,
        "seed": args.seed,
        "split": baseline_record["split"],
        "dataset_sha256": baseline_record.get("dataset_sha256"),
        "dataset_identity_verified": baseline_record.get("dataset_sha256") is not None,
        "source_sha256": {
            "baseline": hashlib.sha256(baseline_bytes).hexdigest(),
            "candidate": hashlib.sha256(candidate_bytes).hexdigest(),
        },
        "baseline_mean": sum(baseline.values()) / len(baseline),
        "candidate_mean": sum(candidate.values()) / len(candidate),
        "query_changes": paired_query_changes(baseline, candidate),
        **result,
    }

    output = args.out / f"{args.baseline.stem}_vs_{args.candidate.stem}_{args.metric}.json"
    write_result(record, output)

    print(f"wrote {output}")
    print(
        f"{args.metric}: {record['baseline_mean']:.4f} -> {record['candidate_mean']:.4f}, "
        f"{format_difference(result)}"
    )
    print("significant" if result["significant"] else "NOT significant at this level")


if __name__ == "__main__":
    main()
