import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts import ann_sweep
from scripts.ann_sweep import best_settings, load_manifest, main, time_search

from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES, checkpoint_hashes
from quant_retrieval.retrieval.index_artifacts import publish_index


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
    np.save(tmp_path / "embeddings_fp32.npy", np.eye(3, dtype=np.float32)[:2])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(tmp_path, checkpoint) == manifest
    with pytest.raises(ValueError, match="checkpoint"):
        load_manifest(tmp_path, tmp_path / "other")
    for key, value in [("documents", 3), ("dimensions", 4), ("max_length", True),
                       ("pooling", "unknown")]:
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
                        SimpleNamespace(_encode=lambda texts: np.array([[1., 0.]]), device="cpu"))

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


def test_ann_preflight_rejects_modified_export_before_runtime_work(tmp_path):
    directory = tmp_path / "export"
    checkpoint = tmp_path / "model"
    publish_index(directory, np.array([1, 2]), np.eye(2),
                  {"checkpoint": str(checkpoint), "max_length": 128})
    assert load_manifest(directory, checkpoint)["schema_version"] == 1
    np.save(directory / "answer_ids.npy", [2, 1])
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manifest(directory, checkpoint)


def test_existing_sweep_report_is_preserved_before_loading_runtime(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("previous measurement")
    monkeypatch.setattr("sys.argv", ["ann", "--embeddings", "missing",
                                    "--output", str(output)])
    with pytest.raises(SystemExit):
        main()
    assert output.read_text() == "previous measurement"


@pytest.mark.parametrize("name", ["manifest.json", "embeddings_fp32.npy", "answer_ids.npy"])
def test_overwrite_cannot_target_benchmark_inputs(tmp_path, monkeypatch, name):
    monkeypatch.setattr("sys.argv", ["ann", "--embeddings", str(tmp_path),
                                    "--output", str(tmp_path / name), "--overwrite"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("options", [
    ["--seed", "-1"], ["--seed", str(2**32)], ["--ef-search", "16", "16"],
    ["--embeddings", "missing", "./missing"],
])
def test_ambiguous_sweep_settings_fail_before_runtime_import(monkeypatch, options):
    monkeypatch.setattr("sys.argv", ["ann", "--embeddings", "missing", *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("ids,vectors", [
    ([1, 1], np.eye(2, dtype=np.float32)),
    ([1, 2], np.zeros((2, 2), dtype=np.float32)),
    ([1, 2], np.eye(2, dtype=np.float16)),
    ([1, 2], np.full((2, 2), np.nan, dtype=np.float32)),
])
def test_legacy_ann_exports_still_require_valid_ids_and_vectors(tmp_path, ids, vectors):
    import json

    np.save(tmp_path / "answer_ids.npy", ids)
    np.save(tmp_path / "embeddings_fp32.npy", vectors)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "documents": 2, "dimensions": 2, "max_length": 64, "checkpoint": str(tmp_path),
    }))
    with pytest.raises(ValueError):
        load_manifest(tmp_path, tmp_path)


@pytest.mark.parametrize("latencies", [[], [True], [-1.], [float("nan")],
                                      [float("inf")], [[1.]], ["1"]])
def test_latency_summaries_refuse_unusable_measurements(latencies):
    with pytest.raises(ValueError, match="latencies"):
        ann_sweep.summarise(latencies)


@pytest.mark.parametrize("settings", [{"repeats": True}, {"warmup": 0.5}, {"k": False}])
def test_timing_counts_must_be_integers(settings):
    with pytest.raises(ValueError):
        time_search(None, np.eye(2), **{"k": 1, **settings})


def test_ann_checks_model_content_even_when_checkpoint_path_is_unchanged(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in CHECKPOINT_FILES:
        (model / name).write_text(name)
    root = tmp_path / "export"
    publish_index(root, np.array([1, 2]), np.eye(2), {
        "max_length": 64, "checkpoint": "/old/machine/model",
        "checkpoint_sha256": checkpoint_hashes(model),
    })
    assert load_manifest(root, model)["documents"] == 2
    (model / "config.json").write_text("changed configuration")
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        load_manifest(root, model)
