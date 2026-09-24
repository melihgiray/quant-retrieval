"""ANN input checks that do not require the FAISS runtime."""

import numpy as np
import pytest

from quant_retrieval.retrieval.ann import ApproximateRetriever


def test_construction_breadth_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="ef_construction"):
        ApproximateRetriever(tmp_path / "vectors.npy", ef_construction=0)


@pytest.mark.parametrize(
    ("vectors", "ids", "message"),
    [
        (np.array([1.0, 2.0]), [1, 2], "two-dimensional"),
        (np.empty((2, 0)), [1, 2], "two-dimensional"),
        (np.ones((2, 2)), [], "empty corpus"),
        (np.ones((2, 2)), [1, 1], "unique"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), [1, 2], "finite"),
        (np.ones((2, 2), dtype=np.int64), [1, 2], "floating point"),
        (np.zeros((2, 2)), [1, 2], "zero rows"),
    ],
)
def test_bad_ann_inputs_fail_before_building_a_graph(tmp_path, vectors, ids, message):
    path = tmp_path / "vectors.npy"
    np.save(path, vectors)

    with pytest.raises(ValueError, match=message):
        ApproximateRetriever(path).index(ids, ["unused"] * len(ids))


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (np.ones((1, 2)), "dimensions"),
        (np.ones(3), "dimensions"),
        (np.array([1.0, np.inf]), "finite"),
        (np.zeros(2), "zero"),
        (np.array([1j, 1.0]), "finite"),
    ],
)
def test_bad_ann_queries_fail_before_faiss_search(tmp_path, query, message):
    retriever = ApproximateRetriever(tmp_path / "vectors.npy")
    retriever._index = object()
    retriever._dimensions = 2
    retriever.document_ids = np.array([1, 2])

    with pytest.raises(ValueError, match=message):
        retriever.search_vector(query, 2)
