import pytest

from quant_retrieval.eval.analysis import error_report


def record():
    return {"split": "val", "run_name": "tiny",
            "per_query": {"2": {"ndcg_at_10": 0.5}, "1": {"ndcg_at_10": 0}},
            "diagnostics": {"2": {"status": "top_k"}, "1": {"status": "not_retrieved"}}}


def test_error_report_orders_worst_queries_and_counts_failures():
    report = error_report(record(), "ndcg_at_10", 1)
    assert report["queries"] == 2
    assert report["worst_queries"][0]["question_id"] == 1
    assert report["status_counts"] == {"not_retrieved": 1, "top_k": 1}


@pytest.mark.parametrize("value", [None, True, float("nan"), -1, 2])
def test_error_report_rejects_invalid_scores(value):
    source = record()
    source["per_query"]["1"]["ndcg_at_10"] = value
    with pytest.raises(ValueError, match="metric values"):
        error_report(source, "ndcg_at_10")


def test_error_report_rejects_missing_diagnostics_and_held_out_runs():
    source = record()
    source["diagnostics"].pop("1")
    with pytest.raises(ValueError, match="same questions"):
        error_report(source, "ndcg_at_10")
    source["split"] = "test"
    with pytest.raises(ValueError, match="restricted"):
        error_report(source, "ndcg_at_10")
