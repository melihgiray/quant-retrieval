import numpy as np
import pandas as pd
import pytest

from quant_retrieval.retrieval.index_artifacts import corpus_ids


@pytest.mark.parametrize("ids", [[0], [-1], [True], [1.5], [None], [1, 1], [2**63]])
def test_export_rejects_lossy_or_ambiguous_document_ids(ids):
    corpus = pd.DataFrame({"answer_id": ids, "text": ["answer"] * len(ids)})
    with pytest.raises(ValueError, match="IDs"):
        corpus_ids(corpus)


@pytest.mark.parametrize("text", [None, 4, " "])
def test_export_rejects_unusable_document_text(text):
    with pytest.raises(ValueError, match="text"):
        corpus_ids(pd.DataFrame({"answer_id": [1], "text": [text]}))


def test_export_preserves_document_order_and_int64_identity():
    corpus = pd.DataFrame({"answer_id": [3, 1, 2], "text": ["a", "b", "c"]})
    ids = corpus_ids(corpus)
    assert ids.dtype == np.int64
    assert ids.tolist() == [3, 1, 2]
