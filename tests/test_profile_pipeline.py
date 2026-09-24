import json
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.profile_pipeline import Stopwatch, instrument, main, run_queries

from quant_retrieval.retrieval.hybrid import HybridRetriever


def test_profiler_excludes_warmup_and_measures_each_repeat():
    calls = []
    retriever = SimpleNamespace(search=lambda query, k: calls.append(query))
    watch = Stopwatch()
    with instrument(retriever, watch, "root"):
        run_queries(retriever, pd.DataFrame({"text": ["a", "b"]}), 10, watch, 3, 2)
    assert calls == ["a", "b", "a", "a", "b", "a", "b"]
    assert watch.report()["root"]["calls"] == 4


@pytest.mark.parametrize("warmup,repeats", [(-1, 1), (0, 0)])
def test_profiler_rejects_invalid_measurement_counts(warmup, repeats):
    with pytest.raises(ValueError):
        run_queries(None, pd.DataFrame({"text": ["a"]}), 1, Stopwatch(), warmup, repeats)


def test_instrumentation_restores_methods_even_after_failure():
    child = SimpleNamespace(search=lambda *args: [])
    original = child.search
    parent = HybridRetriever([child, child])
    watch = Stopwatch()
    with pytest.raises(RuntimeError):
        with instrument(parent, watch, "root"):
            parent.search("query", 1)
            raise RuntimeError("stop")
    assert child.search is original
    assert "search" not in vars(parent)
    child_rows = [value for key, value in watch.report().items() if "child" in key]
    assert len(child_rows) == 1
    assert child_rows[0]["calls"] == 2
    with instrument(parent, watch, "again"):
        parent.search("query", 1)
    assert watch.report()["again"]["calls"] == 1


def test_profile_cli_saves_sample_and_configuration(tmp_path, monkeypatch):
    config = tmp_path / "bm25.yaml"
    config.write_text("retriever: bm25\nseed: 17\nmax_results: 1\n")
    pd.DataFrame({"answer_id": [1, 2], "text": ["bond price", "option price"]}).to_parquet(
        tmp_path / "corpus.parquet")
    pd.DataFrame({"query_id": [1, 2], "text": ["bond", "option"],
                  "split": ["val", "test"]}).to_parquet(tmp_path / "queries.parquet")
    output = tmp_path / "profile.json"
    monkeypatch.setattr("sys.argv", ["profile", "--config", str(config), "--data", str(tmp_path),
                                    "--output", str(output), "--warmup", "1", "--repeats", "2"])
    main()
    report = json.loads(output.read_text())
    assert report["query_ids"] == [1]
    assert report["configuration"]["retriever"] == "bm25"
    assert report["split"] == "val"
    assert report["stages"]["bm25"]["calls"] == report["measured_calls"] == 2
