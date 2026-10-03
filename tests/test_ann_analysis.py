from copy import deepcopy

import pytest

from quant_retrieval.eval.ann_analysis import validate_ann_report


def fixture_report():
    exact = {"artifact": "corpus-a", "documents": 100, "index": "exact", "ef_search": None,
             "samples": 2, "p50_ms": 4., "p95_ms": 4., "build_seconds": .1, "recall_at_k": 1.,
             "latencies_ms": [4., 4.], "per_query_recall": [1., 1.]}
    graph = {**exact, "index": "hnsw", "ef_search": 16, "neighbours": 32,
             "p50_ms": 2., "p95_ms": 2., "recall_at_k": .95,
             "latencies_ms": [2., 2.], "per_query_recall": [.9, 1.]}
    return {"schema_version": 1, "complete": True, "scope": "index_search_only",
            "latency_order": "repeat_major_query_minor", "recall_reference": "exact_top_k_overlap",
            "k": 10, "queries": 2, "repeats": 1, "threads": 1, "neighbours": 32,
            "ef_construction": 200, "ef_search": [16], "warmup": 0, "recall_target": .95,
            "question_ids": [10, 20], "query_sha256": "a" * 64,
            "artifacts": [{"directory": "corpus-a", "manifest": {"documents": 100}}],
            "runs": [exact, graph]}


def test_valid_report_is_not_mutated():
    report = fixture_report()
    before = deepcopy(report)
    validate_ann_report(report)
    assert report == before


@pytest.mark.parametrize("key,value", [
    ("complete", False), ("schema_version", True), ("scope", "end_to_end"),
    ("queries", 0), ("threads", True), ("question_ids", [10, 10]),
    ("query_sha256", "unknown"), ("recall_target", float("nan")),
])
def test_invalid_report_protocol_is_rejected(key, value):
    report = fixture_report()
    report[key] = value
    with pytest.raises(ValueError):
        validate_ann_report(report)


@pytest.mark.parametrize("key,value", [("p50_ms", -1), ("p95_ms", 1),
    ("samples", False), ("recall_at_k", 1.1), ("build_seconds", float("inf"))])
def test_invalid_measurements_are_rejected(key, value):
    report = fixture_report()
    report["runs"][0][key] = value
    with pytest.raises(ValueError):
        validate_ann_report(report)
