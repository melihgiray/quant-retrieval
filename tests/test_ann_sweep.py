import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.ann_sweep import load_manifest, main, time_search


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


def test_manifest_matches_vectors_and_encoder(tmp_path):
    checkpoint = tmp_path / "model"
    manifest = {"documents": 2, "dimensions": 3, "max_length": 128,
                "checkpoint": str(checkpoint)}
    np.save(tmp_path / "answer_ids.npy", [1, 2])
    np.save(tmp_path / "embeddings_fp32.npy", np.ones((2, 3), dtype=np.float32))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(tmp_path, checkpoint) == manifest
    with pytest.raises(ValueError, match="checkpoint"):
        load_manifest(tmp_path, tmp_path / "other")
    for key, value in [("documents", 3), ("dimensions", 4), ("max_length", True)]:
        path.write_text(json.dumps({**manifest, key: value}))
        with pytest.raises(ValueError):
            load_manifest(tmp_path, checkpoint)
