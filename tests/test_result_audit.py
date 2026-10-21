from copy import deepcopy

import pytest

from quant_retrieval.eval.audit import (
    audit_diagnostics,
    audit_metrics,
    audit_protocol,
    audit_provenance,
    audit_rankings,
    validated_scores,
)
from quant_retrieval.eval.diagnostics import ranking_diagnostics
from quant_retrieval.eval.metrics import METRIC_NAMES, aggregate_metrics, per_query_metrics


def result_record():
    return {
        "run_name": "fixture",
        "retriever": "bm25",
        "split": "val",
        "per_query": {
            "20": dict.fromkeys(METRIC_NAMES, 0.6),
            "10": dict.fromkeys(METRIC_NAMES, 0.2),
        },
        "metrics": dict.fromkeys(METRIC_NAMES, 0.4),
        "counts": {"queries": 2, "corpus_documents": 100, "max_results": 100},
    }


def test_score_audit_returns_sorted_independent_rows():
    record = result_record()
    before = deepcopy(record)
    scores = validated_scores(record)
    assert list(scores) == [10, 20]
    scores[10]["ndcg_at_10"] = 0
    assert record == before


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 2])
def test_score_audit_rejects_invalid_values(value):
    record = result_record()
    record["per_query"]["10"]["ndcg_at_10"] = value
    with pytest.raises(ValueError, match="finite numbers"):
        validated_scores(record)


def test_score_audit_requires_the_complete_metric_contract():
    record = result_record()
    del record["per_query"]["20"]["recall_at_100"]
    with pytest.raises(ValueError, match="exactly the supported metrics"):
        validated_scores(record)


def test_metric_audit_recomputes_means_and_tolerates_serialization_rounding():
    record = result_record()
    record["metrics"]["mrr_at_10"] += 0.0000004
    result = audit_metrics(record)
    assert result["recomputed"]["mrr_at_10"] == pytest.approx(0.4)
    assert result["absolute_error"]["mrr_at_10"] == pytest.approx(0.0000004)
    record["metrics"]["mrr_at_10"] = 0.5
    with pytest.raises(ValueError, match="disagree"):
        audit_metrics(record)


@pytest.mark.parametrize("value", [True, None, float("nan"), -1])
def test_metric_audit_rejects_invalid_aggregates(value):
    record = result_record()
    record["metrics"]["ndcg_at_10"] = value
    with pytest.raises(ValueError, match="aggregate metrics must be finite"):
        audit_metrics(record)


@pytest.mark.parametrize(
    "name,value", [("queries", True), ("queries", 3), ("corpus_documents", 0), ("max_results", 10)]
)
def test_protocol_audit_rejects_counts_incompatible_with_metrics(name, value):
    record = result_record()
    record["counts"][name] = value
    with pytest.raises(ValueError):
        audit_protocol(record)


def test_protocol_audit_rejects_held_out_runs_and_preserves_inputs():
    record = result_record()
    assert audit_protocol(record)["counts"] == record["counts"]
    record["split"] = "test"
    with pytest.raises(ValueError, match="train or val"):
        audit_protocol(record)


def test_provenance_audit_distinguishes_recorded_identity_from_reloaded_data():
    record = result_record()
    assert audit_provenance(record)["dataset_identity_recorded"] is False
    record["dataset_sha256"] = dict.fromkeys(("corpus", "queries", "qrels"), "a" * 64)
    record["created_at"] = "2026-10-21T12:00:00+00:00"
    report = audit_provenance(record)
    assert report["dataset_identity_recorded"] is True
    assert report["original_data_reloaded"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_sha256", {}),
        ("commit", "main"),
        ("created_at", "2026-10-21T12:00:00"),
        ("created_at", 17),
        ("created_at", "yesterday"),
    ],
)
def test_provenance_audit_rejects_malformed_evidence(key, value):
    record = {**result_record(), key: value}
    with pytest.raises(ValueError):
        audit_provenance(record)


def diagnostic_record():
    rankings, qrels = {10: [9, 1], 20: [2, 9]}, {10: {1: 2}, 20: {2: 2}}
    return {
        **result_record(),
        "metrics": aggregate_metrics(rankings, qrels),
        "per_query": {str(key): row for key, row in per_query_metrics(rankings, qrels).items()},
        "diagnostics": {str(key): row for key, row in ranking_diagnostics(rankings, qrels).items()},
        "rankings": {str(key): row for key, row in rankings.items()},
    }


def test_diagnostic_audit_cross_checks_metrics_and_reports_missing_evidence():
    assert audit_diagnostics(result_record()) == {"recorded": False}
    record = diagnostic_record()
    assert audit_diagnostics(record)["status_counts"] == {"top_k": 2}
    record["per_query"]["10"]["mrr_at_10"] = 0.3
    with pytest.raises(ValueError, match="MRR disagrees"):
        audit_diagnostics(record)


def test_diagnostic_audit_checks_primary_recall_counts():
    record = diagnostic_record()
    record["per_query"]["10"]["recall_at_100"] = 0.5
    with pytest.raises(ValueError, match="Recall@100 disagrees"):
        audit_diagnostics(record)


def test_ranking_audit_reports_structure_without_claiming_corpus_membership():
    assert audit_rankings(result_record()) == {"recorded": False}
    report = audit_rankings(diagnostic_record())
    assert report["minimum_returned"] == report["maximum_returned"] == 2
    assert report["corpus_membership_verified"] is False


@pytest.mark.parametrize("ranking", [[9, 9], [True, 1], [9], [2**63, 1], "9,1"])
def test_ranking_audit_checks_ids_and_diagnostic_count(ranking):
    record = diagnostic_record()
    record["rankings"]["10"] = ranking
    with pytest.raises(ValueError, match="ranking"):
        audit_rankings(record)
