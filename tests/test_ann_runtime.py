"""Exercise wrapper behavior without loading the platform FAISS runtime."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts import ann_sweep

from quant_retrieval.retrieval.ann import ApproximateRetriever


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
    np.save(tmp_path / "answer_ids.npy", [10, 20])
    np.save(tmp_path / "embeddings_fp32.npy", np.eye(2, dtype=np.float32))
    checkpoint = tmp_path / "model"
    (tmp_path / "manifest.json").write_text(json.dumps({
        "documents": 2, "dimensions": 2, "max_length": 64, "checkpoint": str(checkpoint),
    }))
    pd.DataFrame({"question_id": [1, 2], "text": ["one", "two"], "split": ["val"] * 2
                  }).to_parquet(tmp_path / "queries.parquet")
    encoder_settings = []

    def encoder(*args, **kwargs):
        encoder_settings.append(kwargs)
        return SimpleNamespace(_encode=lambda texts: np.eye(2, dtype=np.float32), device="cpu")

    monkeypatch.setattr(ann_sweep, "DenseRetriever", encoder)
    output = tmp_path / "sweep.json"
    monkeypatch.setattr("sys.argv", ["ann", "--embeddings", str(tmp_path), "--data", str(tmp_path),
        "--checkpoint", str(checkpoint), "--output", str(output), "--ef-search", "16", "32",
        "--warmup", "1", "--repeats", "2", "--threads", "2", "--k", "1"])
    ann_sweep.main()
    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert len(created) == 2
    assert [index.calls for index in created] == [5, 10]
    assert created[1].hnsw.efSearch == 32
    assert threads == [2]
    assert encoder_settings[0]["max_length"] == 64
    assert len(report["runs"]) == 3
    assert all(row["samples"] == 4 and row["recall_at_k"] == 1 for row in report["runs"])
    assert len(report["question_ids"]) == 2
    assert report["summary"][0]["best"]["index"] == "hnsw"
