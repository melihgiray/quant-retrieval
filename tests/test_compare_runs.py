import json

import pytest
from scripts.compare_runs import load_per_query


@pytest.mark.parametrize("query_id", ["01", "-1", "0", "1.0", "one"])
def test_comparison_rejects_ids_that_cannot_be_paired_unambiguously(tmp_path, query_id):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {query_id: {"ndcg_at_10": 0.5}}}))
    with pytest.raises(SystemExit, match="invalid query ID"):
        load_per_query(path, "ndcg_at_10")


@pytest.mark.parametrize("value", [None, True, "0.5", -0.1, 1.1, float("nan")])
def test_comparison_rejects_invalid_metric_values(tmp_path, value):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {"1": {"ndcg_at_10": value}}}))
    with pytest.raises(SystemExit, match="invalid.*score"):
        load_per_query(path, "ndcg_at_10")


def test_comparison_loads_valid_query_scores(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {"12": {"ndcg_at_10": 0.5}}}))
    assert load_per_query(path, "ndcg_at_10") == {12: 0.5}
