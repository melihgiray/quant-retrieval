"""Offline checks and operating-point analysis for versioned ANN measurements."""

import math
import re


def _integer(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, maximum: float = math.inf) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= maximum):
        raise ValueError(f"{name} must be a finite number between 0 and {maximum}")


def validate_ann_report(report: dict) -> None:
    """Reject partial runs and protocols whose timings are not comparable."""
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int:
        raise ValueError("a versioned ANN report is required")
    if report["schema_version"] != 1 or report.get("complete") is not True:
        raise ValueError("a completed schema version 1 ANN report is required")
    for name, expected in {"scope": "index_search_only",
                           "latency_order": "repeat_major_query_minor",
                           "recall_reference": "exact_top_k_overlap"}.items():
        if report.get(name) != expected:
            raise ValueError(f"unsupported ANN {name}")
    for name in ("k", "queries", "repeats", "threads", "neighbours", "ef_construction"):
        _integer(report.get(name), name)
    _integer(report.get("warmup"), "warmup", 0)
    _number(report.get("recall_target"), "recall_target", 1)
    ids = report.get("question_ids")
    if not isinstance(ids, list) or len(ids) != report["queries"]:
        raise ValueError("question IDs must match the query count")
    for value in ids:
        _integer(value, "question ID")
    if len(set(ids)) != len(ids):
        raise ValueError("question IDs must be unique")
    fingerprint = report.get("query_sha256")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("query_sha256 must identify the sampled queries")
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("ANN runs must be a nonempty list")
    for row in runs:
        if not isinstance(row, dict) or row.get("index") not in {"exact", "hnsw"}:
            raise ValueError("unknown ANN index type")
        if not isinstance(row.get("artifact"), str) or not row["artifact"].strip():
            raise ValueError("each run must name its artifact")
        for name in ("documents", "samples"):
            _integer(row.get(name), name)
        for name in ("p50_ms", "p95_ms", "build_seconds"):
            _number(row.get(name), name)
        _number(row.get("recall_at_k"), "recall_at_k", 1)
        if row["p95_ms"] < row["p50_ms"]:
            raise ValueError("p95 latency cannot be below p50")
        if row["index"] == "hnsw":
            _integer(row.get("ef_search"), "ef_search")
        elif row.get("ef_search") is not None or row["recall_at_k"] != 1:
            raise ValueError("exact rows must have unit recall and no search breadth")
