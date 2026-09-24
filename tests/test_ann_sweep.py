from types import SimpleNamespace

import numpy as np
import pytest
from scripts.ann_sweep import main, time_search


def test_warmup_is_excluded_from_results_and_timings():
    calls = []
    retriever = SimpleNamespace(search_vector=lambda vector, k: calls.append(vector[0]))
    results, timings = time_search(retriever, np.array([[1], [2]]), 1, warmup=3)
    assert calls == [1, 2, 1, 1, 2]
    assert len(results) == len(timings) == 2


@pytest.mark.parametrize("queries,warmup", [(np.empty((0, 2)), 0), (np.ones((1, 2)), -1)])
def test_unusable_timing_inputs(queries, warmup):
    with pytest.raises(ValueError):
        time_search(None, queries, 1, warmup)


@pytest.mark.parametrize("option,value", [
    ("--threads", "0"), ("--ef-construction", "-1"), ("--ef-search", "0"),
    ("--queries", "0"), ("--warmup", "-1"), ("--k", "0"),
])
def test_invalid_cli_settings_fail_before_loading_runtime(monkeypatch, option, value):
    monkeypatch.setattr("sys.argv", ["ann_sweep", "--embeddings", "missing", option, value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
