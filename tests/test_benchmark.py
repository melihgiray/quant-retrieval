import pandas as pd

from quant_retrieval.eval.benchmark import benchmark_context


def test_benchmark_context_tracks_query_content_and_order(monkeypatch):
    monkeypatch.setattr("quant_retrieval.eval.benchmark.current_commit", lambda: "abc1234")
    queries = pd.DataFrame({"query_id": [1, 2], "text": ["one", "two"], "split": ["val"] * 2})
    result = benchmark_context(queries, 17)
    assert result["query_ids"] == [1, 2]
    assert result["seed"] == 17
    assert result["commit"] == "abc1234"
    assert result["environment"]["python"]
    assert result["query_sha256"] == benchmark_context(queries.copy(), 17)["query_sha256"]
    assert result["query_sha256"] != benchmark_context(queries.iloc[::-1], 17)["query_sha256"]
    queries.loc[0, "text"] = "changed"
    assert result["query_sha256"] != benchmark_context(queries, 17)["query_sha256"]
