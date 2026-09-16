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
    ],
)
def test_bad_ann_inputs_fail_before_building_a_graph(tmp_path, vectors, ids, message):
    path = tmp_path / "vectors.npy"
    np.save(path, vectors)

    with pytest.raises(ValueError, match=message):
        ApproximateRetriever(path).index(ids, ["unused"] * len(ids))
