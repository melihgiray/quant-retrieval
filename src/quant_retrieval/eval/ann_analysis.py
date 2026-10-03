"""Offline checks and operating-point analysis for versioned ANN measurements."""

import math
import re

import numpy as np


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
        if not isinstance(row, dict) or row.get("index") not in ("exact", "hnsw"):
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
    _validate_coverage(report)


def _validate_coverage(report: dict) -> None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact declarations are required")
    documents = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("manifest"), dict):
            raise ValueError("each artifact needs a manifest")
        directory = artifact.get("directory")
        if not isinstance(directory, str) or not directory.strip() or directory in documents:
            raise ValueError("artifact directories must be nonempty and unique")
        count = artifact["manifest"].get("documents")
        _integer(count, "manifest documents")
        documents[directory] = count
    settings = report.get("ef_search")
    if not isinstance(settings, list) or not settings:
        raise ValueError("search settings are required")
    for value in settings:
        _integer(value, "ef_search")
    if len(set(settings)) != len(settings):
        raise ValueError("search settings must be unique")
    expected = {(directory, index, breadth) for directory in documents
                for index, breadth in [("exact", None), *[("hnsw", n) for n in settings]]}
    seen = set()
    for row in report["runs"]:
        key = (row["artifact"], row["index"], row.get("ef_search"))
        if key not in expected or key in seen:
            raise ValueError("duplicate or undeclared ANN measurement")
        seen.add(key)
        if row["documents"] != documents[row["artifact"]]:
            raise ValueError("run document count disagrees with its artifact")
        if row["index"] == "hnsw":
            _integer(row.get("neighbours"), "neighbours")
            if row["neighbours"] != report["neighbours"]:
                raise ValueError("graph neighbours disagree with the sweep settings")
        count = report["queries"] * report["repeats"]
        latencies, recalls = row.get("latencies_ms"), row.get("per_query_recall")
        if (row["samples"] != count or not isinstance(latencies, list)
                or len(latencies) != count or not isinstance(recalls, list)
                or len(recalls) != report["queries"]):
            raise ValueError("raw sample counts disagree with the timing protocol")
        for value in latencies:
            _number(value, "latency sample")
        for value in recalls:
            _number(value, "query recall", 1)
        derived = {"p50_ms": float(np.percentile(latencies, 50)),
                   "p95_ms": float(np.percentile(latencies, 95)),
                   "recall_at_k": math.fsum(recalls) / len(recalls)}
        if any(not math.isclose(row[name], value, rel_tol=1e-12, abs_tol=1e-12)
               for name, value in derived.items()):
            raise ValueError("saved aggregates disagree with raw measurements")
    if seen != expected:
        raise ValueError("completed report is missing declared sweep points")


def analyze_ann_report(report: dict, recall_target: float | None = None) -> dict:
    """Choose measured operating points, without claiming significance or a crossover."""
    validate_ann_report(report)
    target = report["recall_target"] if recall_target is None else recall_target
    _number(target, "recall target", 1)
    summaries = []
    fields = ("index", "ef_search", "recall_at_k", "p50_ms", "p95_ms", "build_seconds")
    for artifact in report["artifacts"]:
        group = [row for row in report["runs"] if row["artifact"] == artifact["directory"]]
        exact = next(row for row in group if row["index"] == "exact")
        eligible = [row for row in group if row["index"] == "hnsw" and row["recall_at_k"] >= target]
        best = min(eligible, key=lambda row: (row["p50_ms"], row["p95_ms"], row["ef_search"]),
                   default=None)
        frontier = [row for row in group if not any(
            other["p50_ms"] <= row["p50_ms"] and other["recall_at_k"] >= row["recall_at_k"]
            and (other["p50_ms"] < row["p50_ms"] or other["recall_at_k"] > row["recall_at_k"])
            for other in group
        )]
        faster = best is not None and best["p50_ms"] < exact["p50_ms"]
        summaries.append({
            "artifact": artifact["directory"], "documents": exact["documents"],
            "exact": {name: exact[name] for name in fields},
            "best_eligible_hnsw": None if best is None else {name: best[name] for name in fields},
            "observed_winner": "hnsw" if faster else "exact",
            "speedup": (exact["p50_ms"] / best["p50_ms"]
                        if best is not None and best["p50_ms"] > 0 else None),
            "pareto_frontier": [{name: row[name] for name in fields} for row in sorted(
                frontier, key=lambda row: (row["p50_ms"], -row["recall_at_k"],
                                          row["ef_search"] or 0))],
        })
    return {"scope": "index_search_only", "recall_target": target, "artifacts": summaries,
            "interpretation": "Observed p50 comparisons, not significance or relevance quality."}
