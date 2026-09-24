"""Exercise wrapper behavior without loading the platform FAISS runtime."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

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
