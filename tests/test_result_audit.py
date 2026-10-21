from copy import deepcopy

import pytest

from quant_retrieval.eval.audit import audit_metrics, validated_scores
from quant_retrieval.eval.metrics import METRIC_NAMES


def result_record():
    return {"run_name": "fixture", "retriever": "bm25", "split": "val",
            "per_query": {"20": dict.fromkeys(METRIC_NAMES, .6),
                          "10": dict.fromkeys(METRIC_NAMES, .2)},
            "metrics": dict.fromkeys(METRIC_NAMES, .4),
            "counts": {"queries": 2, "corpus_documents": 100, "max_results": 100}}


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
    record["metrics"]["mrr_at_10"] += .0000004
    result = audit_metrics(record)
    assert result["recomputed"]["mrr_at_10"] == pytest.approx(.4)
    assert result["absolute_error"]["mrr_at_10"] == pytest.approx(.0000004)
    record["metrics"]["mrr_at_10"] = .5
    with pytest.raises(ValueError, match="disagree"):
        audit_metrics(record)


@pytest.mark.parametrize("value", [True, None, float("nan"), -1])
def test_metric_audit_rejects_invalid_aggregates(value):
    record = result_record()
    record["metrics"]["ndcg_at_10"] = value
    with pytest.raises(ValueError, match="aggregate metrics must be finite"):
        audit_metrics(record)
