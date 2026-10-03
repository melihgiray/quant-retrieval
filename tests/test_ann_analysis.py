import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from scripts import analyze_ann

from quant_retrieval.eval.ann_analysis import (
    analyze_ann_report,
    query_diagnostics,
    validate_ann_report,
)


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


@pytest.mark.parametrize("change", ["missing", "duplicate", "count", "latency", "recall",
                                    "samples", "undeclared", "graph"])
def test_complete_reports_must_cover_the_declared_sweep_and_match_raw_data(change):
    report = fixture_report()
    if change == "missing":
        report["runs"].pop()
    elif change == "duplicate":
        report["runs"].append(deepcopy(report["runs"][0]))
    elif change == "count":
        report["runs"][0]["documents"] += 1
    elif change == "latency":
        report["runs"][0]["latencies_ms"][0] = 8.
    elif change == "recall":
        report["runs"][1]["per_query_recall"][0] = .1
    elif change == "samples":
        report["repeats"] = 2
    elif change == "undeclared":
        report["runs"][1]["ef_search"] = 32
    else:
        report["runs"][1]["neighbours"] = 16
    with pytest.raises(ValueError):
        validate_ann_report(report)


def test_operating_points_honor_recall_and_do_not_trust_saved_summaries():
    report = fixture_report()
    report["summary"] = "stale summary must be ignored"
    point = analyze_ann_report(report)["artifacts"][0]
    assert point["observed_winner"] == "hnsw"
    assert point["speedup"] == 2
    assert len(point["pareto_frontier"]) == 2
    point = analyze_ann_report(report, 1.)["artifacts"][0]
    assert point["observed_winner"] == "exact"
    assert point["best_eligible_hnsw"] is None


def test_dominated_hnsw_point_is_removed_from_frontier():
    report = fixture_report()
    slower = {**report["runs"][1], "ef_search": 32, "p50_ms": 3., "p95_ms": 3.,
              "latencies_ms": [3., 3.]}
    report["runs"].append(slower)
    report["ef_search"].append(32)
    point = analyze_ann_report(report)["artifacts"][0]
    assert point["best_eligible_hnsw"]["ef_search"] == 16
    assert [row["ef_search"] for row in point["pareto_frontier"]] == [16, None]


def test_equal_size_corpora_keep_separate_operating_points():
    report = fixture_report()
    report["artifacts"].append({"directory": "corpus-b", "manifest": {"documents": 100}})
    other = deepcopy(report["runs"])
    for row in other:
        row["artifact"] = "corpus-b"
    other[0].update(p50_ms=1., p95_ms=1., latencies_ms=[1., 1.])
    report["runs"].extend(other)
    assert [row["observed_winner"] for row in analyze_ann_report(report)["artifacts"]] == [
        "hnsw", "exact"]


def test_analysis_cli_preserves_source_identity_and_protects_existing_files(tmp_path, monkeypatch):
    source, output = tmp_path / "sweep.json", tmp_path / "analysis.json"
    source.write_text(json.dumps(fixture_report()))
    original = source.read_bytes()
    monkeypatch.setattr("sys.argv", ["analyze", "--report", str(source),
                                    "--output", str(output), "--recall-target", "1"])
    analyze_ann.main()
    result = json.loads(output.read_text())
    assert result["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert result["source"]["benchmark"]["question_ids"] == [10, 20]
    assert result["artifacts"][0]["observed_winner"] == "exact"
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        analyze_ann.main()
    assert output.read_bytes() == saved
    monkeypatch.setattr("sys.argv", ["analyze", "--report", str(source), "--output", str(source)])
    with pytest.raises(SystemExit):
        analyze_ann.main()
    assert source.read_bytes() == original


def test_query_diagnostics_align_repeated_samples_with_question_identity():
    report = fixture_report()
    report["repeats"] = 2
    for row, values in zip(report["runs"], ([2., 8., 4., 10.], [1., 7., 3., 9.]), strict=True):
        row.update(latencies_ms=values, samples=4, p50_ms=float(np.percentile(values, 50)),
                   p95_ms=float(np.percentile(values, 95)))
    details = query_diagnostics(report, limit=1)[0]["worst_queries"]
    assert details == [{"question_id": 10, "recall_at_k": .9,
                        "hnsw_p50_ms": 2., "exact_p50_ms": 3.}]
    with pytest.raises(ValueError, match="limit"):
        query_diagnostics(report, limit=0)
