from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.profile_pipeline import Stopwatch, instrument, run_queries

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
