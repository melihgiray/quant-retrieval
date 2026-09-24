from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.profile_pipeline import Stopwatch, instrument, run_queries


def test_profiler_excludes_warmup_and_measures_each_repeat():
    calls = []
    retriever = SimpleNamespace(search=lambda query, k: calls.append(query))
    watch = Stopwatch()
    instrument(retriever, watch, "root")
    run_queries(retriever, pd.DataFrame({"text": ["a", "b"]}), 10, watch, 3, 2)
    assert calls == ["a", "b", "a", "a", "b", "a", "b"]
    assert watch.report()["root"]["calls"] == 4


@pytest.mark.parametrize("warmup,repeats", [(-1, 1), (0, 0)])
def test_profiler_rejects_invalid_measurement_counts(warmup, repeats):
    with pytest.raises(ValueError):
        run_queries(None, pd.DataFrame({"text": ["a"]}), 1, Stopwatch(), warmup, repeats)
