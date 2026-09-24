from types import SimpleNamespace

import numpy as np
import pytest
from scripts.ann_sweep import time_search


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
