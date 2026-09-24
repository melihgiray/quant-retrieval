"""Offline summaries of saved per-query evaluation evidence."""

import math
from collections import Counter

from quant_retrieval.eval.metrics import METRIC_NAMES


def paired_query_changes(baseline: dict, candidate: dict, limit: int = 10) -> dict:
    """Locate observed wins and losses; this is not a significance test."""
    if not baseline or set(baseline) != set(candidate) or limit <= 0:
        raise ValueError("paired changes need identical nonempty questions and a positive limit")
    for value in [*baseline.values(), *candidate.values()]:
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("paired scores must be finite and between zero and one")
    rows = [{"question_id": key, "baseline": baseline[key], "candidate": candidate[key],
             "delta": candidate[key] - baseline[key]} for key in sorted(baseline)]
    wins = sorted([row for row in rows if row["delta"] > 0],
                  key=lambda row: (-row["delta"], row["question_id"]))
    losses = sorted([row for row in rows if row["delta"] < 0],
                    key=lambda row: (row["delta"], row["question_id"]))
    return {"improved": len(wins), "regressed": len(losses),
            "unchanged": len(rows) - len(wins) - len(losses),
            "largest_improvements": wins[:limit], "largest_regressions": losses[:limit]}


def error_report(record: dict, metric: str, limit: int = 20) -> dict:
    if metric not in METRIC_NAMES or limit <= 0:
        raise ValueError("choose a supported metric and a positive example limit")
    if record.get("split") not in {"train", "val"}:
        raise ValueError("error analysis is restricted to train and val runs")
    scores, diagnostics = record.get("per_query"), record.get("diagnostics")
    if not isinstance(scores, dict) or not scores or not isinstance(diagnostics, dict):
        raise ValueError("run must contain per-query scores and diagnostics")
    if set(scores) != set(diagnostics):
        raise ValueError("diagnostics and scores must cover the same questions")
    rows = []
    statuses = {"top_k", "below_cutoff", "not_retrieved", "no_primary_label"}
    for key, values in scores.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or int(key) <= 0:
            raise ValueError("question IDs must be positive integer strings")
        if str(int(key)) != key:
            raise ValueError("question IDs must use canonical integer spelling")
        value = values.get(metric) if isinstance(values, dict) else None
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("per-query metric values must be finite and between zero and one")
        detail = diagnostics[key]
        if not isinstance(detail, dict) or detail.get("status") not in statuses:
            raise ValueError("unknown retrieval diagnostic status")
        rows.append({**detail, "question_id": int(key), "score": value})
    rows.sort(key=lambda row: (row["score"], row["question_id"]))
    return {
        "run_name": record.get("run_name"), "split": record["split"], "metric": metric,
        "queries": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "worst_queries": rows[:limit], "dataset_sha256": record.get("dataset_sha256"),
    }
