"""Exercise wrapper behavior without loading the platform FAISS runtime."""

import json
import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts import ann_sweep

from quant_retrieval.retrieval.ann import ApproximateRetriever
from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES, checkpoint_hashes
from quant_retrieval.retrieval.index_artifacts import publish_index


class FakeIndex:
    def __init__(self, *args):
        self.hnsw = SimpleNamespace()

    def add(self, vectors):
        self.vectors = vectors.copy()

    def search(self, query, k):
        scores = query @ self.vectors.T
        positions = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, positions, axis=1), positions


@pytest.fixture
def fake_faiss(monkeypatch):
    def normalize(vectors):
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    module = SimpleNamespace(normalize_L2=normalize, IndexFlatIP=FakeIndex,
                             IndexHNSWFlat=FakeIndex, METRIC_INNER_PRODUCT=0)
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def test_ann_search_does_not_normalize_the_callers_query(tmp_path, fake_faiss):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2, dtype=np.float32))
    retriever = ApproximateRetriever(path, exact=True)
    retriever.index([10, 20], [])
    query = np.array([3.0, 4.0], dtype=np.float32)
    query.flags.writeable = False
    assert retriever.search_vector(query, 1)[0].document_id == 20
    np.testing.assert_array_equal(query, [3.0, 4.0])


def test_failed_ann_rebuild_preserves_the_previous_index(tmp_path, fake_faiss):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2, dtype=np.float32))
    retriever = ApproximateRetriever(path, exact=True)
    retriever.index([10, 20], [])

    class BrokenIndex(FakeIndex):
        def add(self, vectors):
            raise RuntimeError("allocation failed")

    fake_faiss.IndexFlatIP = BrokenIndex
    with pytest.raises(RuntimeError, match="allocation failed"):
        retriever.index([30, 40], [])
    assert retriever.search_vector(np.array([1.0, 0.0]), 1)[0].document_id == 10


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_ann_normalization_handles_extreme_finite_vectors(tmp_path, fake_faiss, magnitude):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2) * magnitude)
    retriever = ApproximateRetriever(path, exact=True)
    retriever.index([10, 20], [])
    hits = retriever.search_vector(np.array([magnitude, 0.0]), 1)
    assert hits[0].document_id == 10
    assert hits[0].score == pytest.approx(1.0)


def test_search_breadth_changes_without_rebuilding_the_graph(tmp_path, fake_faiss):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2, dtype=np.float32))
    retriever = ApproximateRetriever(path)
    retriever.index([10, 20], [])
    graph = retriever._index
    retriever.set_ef_search(200)
    assert retriever._index is graph
    assert graph.hnsw.efSearch == retriever.ef_search == 200
    assert retriever.search_vector(np.array([1.0, 0.0]), 1)[0].document_id == 10


def test_complete_sweep_reuses_graph_and_saves_reproducible_report(
    tmp_path, fake_faiss, monkeypatch
):
    created, threads = [], []
    wrappers = []

    class TrackingRetriever(ApproximateRetriever):
        def __init__(self, *args, **kwargs):
            assert all(reference() is None for reference in wrappers)
            super().__init__(*args, **kwargs)
            wrappers.append(weakref.ref(self))

    monkeypatch.setattr(ann_sweep, "ApproximateRetriever", TrackingRetriever)

    class CountingIndex(FakeIndex):
        def __init__(self, *args):
            super().__init__(*args)
            self.calls = 0
            created.append(self)

        def search(self, query, k):
            self.calls += 1
            return super().search(query, k)

    fake_faiss.IndexFlatIP = fake_faiss.IndexHNSWFlat = CountingIndex
    fake_faiss.omp_set_num_threads = threads.append
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    for name in CHECKPOINT_FILES:
        (checkpoint / name).write_text(name)
    artifact = tmp_path / "export"
    publish_index(artifact, np.array([10, 20]), np.eye(2, dtype=np.float32), {
        "documents": 2, "dimensions": 2, "max_length": 64, "checkpoint": str(checkpoint),
        "pooling": "cls", "checkpoint_sha256": checkpoint_hashes(checkpoint),
    })
    pd.DataFrame({"question_id": [1, 2], "text": ["one", "two"], "split": ["val"] * 2
                  }).to_parquet(tmp_path / "queries.parquet")
    encoder_settings = []

    def encoder(*args, **kwargs):
        encoder_settings.append(kwargs)
        return SimpleNamespace(_encode=lambda texts: np.eye(2, dtype=np.float32), device="cpu")

    monkeypatch.setattr(ann_sweep, "DenseRetriever", encoder)
    output = tmp_path / "sweep.json"
    monkeypatch.setattr("sys.argv", ["ann", "--embeddings", str(artifact), "--data", str(tmp_path),
        "--checkpoint", str(checkpoint), "--output", str(output), "--ef-search", "16", "32",
        "--warmup", "1", "--repeats", "2", "--threads", "2", "--k", "1"])
    ann_sweep.main()
    report = json.loads(output.read_text())
    assert all(reference() is None for reference in wrappers)
    assert report["complete"] is True
    assert len(created) == 2
    assert [index.calls for index in created] == [5, 10]
    assert created[1].hnsw.efSearch == 32
    assert threads == [2]
    assert encoder_settings[0]["max_length"] == 64
    assert encoder_settings[0]["pooling"] == "cls"
    assert len(report["runs"]) == 3
    assert all(row["samples"] == 4 and row["recall_at_k"] == 1 for row in report["runs"])
    assert len(report["question_ids"]) == 2
    assert report["summary"][0]["best"]["index"] == "hnsw"
    assert report["scope"] == "index_search_only"
    assert report["latency_order"] == "repeat_major_query_minor"
    assert report["artifacts"][0]["payload_checksums_verified"]
    assert report["artifacts"][0]["checkpoint_checksums_verified"]
    for row in report["runs"]:
        assert row["build_seconds"] >= 0
        assert len(row["latencies_ms"]) == 4
        assert row["p50_ms"] == pytest.approx(np.percentile(row["latencies_ms"], 50))
        assert row["per_query_recall"] == [1., 1.]
    analysis_path = tmp_path / "analysis.json"
    # A fresh process proves that report analysis does not load either native runtime.
    completed = subprocess.run([
        sys.executable, "-c",
        "import runpy, sys; runpy.run_module('scripts.analyze_ann', run_name='__main__'); "
        "assert 'torch' not in sys.modules; assert 'faiss' not in sys.modules",
        "--report", str(output), "--output", str(analysis_path),
    ], capture_output=True, text=True, check=True)
    assert "audited 1 artifacts" in completed.stdout
    analysis = json.loads(analysis_path.read_text())
    assert analysis["source"]["benchmark"]["question_ids"] == report["question_ids"]
    assert len(analysis["query_diagnostics"]) == 2
    assert all(len(row["worst_queries"]) == 2 for row in analysis["query_diagnostics"])
