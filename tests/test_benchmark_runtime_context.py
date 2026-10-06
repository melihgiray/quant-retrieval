import pandas as pd

from quant_retrieval.eval import benchmark


def test_measurement_context_is_independent_of_query_encoding_machine(monkeypatch):
    monkeypatch.setattr(benchmark, "current_commit", lambda: "first")
    queries = pd.DataFrame({"question_id": [10], "text": ["query"], "split": ["val"]})
    original = benchmark.benchmark_context(queries, 17)
    monkeypatch.setattr(benchmark, "current_commit", lambda: "second")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "other-cpu")
    new = benchmark.runtime_context()
    assert new["commit"] == "second"
    assert new["environment"]["machine"] == "other-cpu"
    assert "question_ids" not in new and "seed" not in new
    assert benchmark.benchmark_context(queries, 17)["query_sha256"] == original["query_sha256"]
    assert original["commit"] == "first"
