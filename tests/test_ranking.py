import numpy as np
import pytest

from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.ranking import top_k_rows


@pytest.mark.parametrize("ids", [[10, 20, 30, 40, 50], [50, 20, 40, 10, 30]])
def test_dense_cutoff_ties_choose_lowest_document_ids(ids, monkeypatch):
    retriever = DenseRetriever("unused", device="cpu")
    retriever.document_ids = np.array(ids)
    retriever.embeddings = np.tile([1., 0.], (5, 1))
    monkeypatch.setattr(retriever, "_encode", lambda texts: np.array([[1., 0.]]))
    assert [hit.document_id for hit in retriever.search("query", 2)] == [10, 20]


def test_top_k_matches_full_sort_with_mixed_ties():
    scores = np.array([0., 1., 1., 2., -1., 1.])
    ids = np.array([9, 8, 7, 6, 5, 4])
    expected = np.lexsort((ids, -scores))
    for k in range(1, 9):
        np.testing.assert_array_equal(top_k_rows(scores, ids, k), expected[:k])
