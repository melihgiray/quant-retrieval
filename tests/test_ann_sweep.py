import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts import ann_sweep
from scripts.ann_sweep import best_settings, load_manifest, main, time_search


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


def test_failed_graph_build_keeps_completed_exact_measurement(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    queries = pd.DataFrame({"question_id": [1], "text": ["query"], "split": ["val"]})
    monkeypatch.setitem(__import__("sys").modules, "faiss",
                        SimpleNamespace(omp_set_num_threads=lambda threads: None))
    monkeypatch.setattr(ann_sweep, "set_seed", lambda seed: None)
    monkeypatch.setattr(ann_sweep, "benchmark_context", lambda *args: {})
    monkeypatch.setattr(ann_sweep, "load_manifest", lambda *args:
                        {"documents": 1, "dimensions": 2, "max_length": 128})
    monkeypatch.setattr(ann_sweep.pd, "read_parquet", lambda path: queries)
    monkeypatch.setattr(ann_sweep.np, "load", lambda path: np.array([1]))
    monkeypatch.setattr(ann_sweep, "DenseRetriever", lambda *args, **kwargs:
                        SimpleNamespace(_encode=lambda texts: np.ones((1, 2)), device="cpu"))

    class Retriever:
        def __init__(self, *args, exact=False, **kwargs):
            self.exact = exact

        def index(self, *args):
            if not self.exact:
                raise RuntimeError("graph build failed")

        def search_vector(self, *args):
            return []

    monkeypatch.setattr(ann_sweep, "ApproximateRetriever", Retriever)
    monkeypatch.setattr("sys.argv", ["ann_sweep", "--embeddings", str(tmp_path),
                                    "--output", str(output)])
    with pytest.raises(RuntimeError, match="graph build failed"):
        main()
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert len(report["runs"]) == 1
    assert report["runs"][0]["index"] == "exact"


def test_summary_never_compares_different_corpora_with_equal_sizes():
    runs = [
        {"artifact": "a", "documents": 10, "index": "exact", "p50_ms": 8},
        {"artifact": "b", "documents": 10, "index": "exact", "p50_ms": 2},
        {"artifact": "a", "documents": 10, "index": "hnsw", "p50_ms": 3, "recall_at_k": .96},
        {"artifact": "b", "documents": 10, "index": "hnsw", "p50_ms": 1, "recall_at_k": .90},
    ]
    result = best_settings(runs, .95)
    assert result[0]["best"]["artifact"] == "a"
    assert result[0]["exact_p50_ms"] == 8
    assert result[1]["best"] is None
    assert best_settings(runs, .9)[1]["best"]["p50_ms"] == 1


def test_ann_repeats_add_timings_without_duplicating_recall_queries():
    calls = []
    retriever = SimpleNamespace(search_vector=lambda vector, k: calls.append(vector[0]))
    results, latencies = time_search(retriever, np.array([[1], [2]]), 1, 1, 3)
    assert len(calls) == 7
    assert len(results) == 2
    assert len(latencies) == 6
    assert ann_sweep.summarise([0.0001, 0.0002])["p50_ms"] > 0
    with pytest.raises(ValueError, match="repeats"):
        time_search(retriever, np.ones((1, 2)), 1, repeats=0)
