import numpy as np
import pandas as pd
import pytest

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.index_artifacts import publish_index
from quant_retrieval.retrieval.precomputed import PrecomputedDenseRetriever


def fixture_export(tmp_path):
    corpus = pd.DataFrame({"answer_id": [3, 1, 2], "text": ["first", "second", "third"]})
    root, checkpoint = tmp_path / "export", tmp_path / "model"
    publish_index(root, corpus.answer_id.to_numpy(), np.eye(3),
        {"max_length": 64, "checkpoint": str(checkpoint),
         "corpus_sha256": corpus_fingerprint(corpus), "pooling": "mean"})
    return corpus, root, checkpoint


@pytest.mark.parametrize("precision,dtype", [("fp16", np.float16), ("fp32", np.float32)])
def test_precomputed_indexes_without_encoding_documents(tmp_path, monkeypatch, precision, dtype):
    corpus, root, checkpoint = fixture_export(tmp_path)
    retriever = PrecomputedDenseRetriever(str(checkpoint), root, precision, device="cpu")
    retriever.index(corpus.answer_id.tolist(), corpus.text.tolist())
    assert isinstance(retriever.embeddings, np.memmap)
    assert retriever.embeddings.dtype == dtype
    assert retriever._model is None
    assert retriever.max_length == 64
    monkeypatch.setattr(retriever, "_encode", lambda texts: np.array([[0., 1., 0.]]))
    assert retriever.search("query", 1)[0].document_id == 1


def test_precomputed_rejects_changed_corpus_without_losing_previous_index(tmp_path):
    corpus, root, checkpoint = fixture_export(tmp_path)
    retriever = PrecomputedDenseRetriever(str(checkpoint), root, device="cpu")
    retriever.index(corpus.answer_id.tolist(), corpus.text.tolist())
    previous = retriever.embeddings
    with pytest.raises(ValueError, match="evaluation corpus"):
        retriever.index(corpus.answer_id.tolist(), ["changed", "second", "third"])
    assert retriever.embeddings is previous
    with pytest.raises(ValueError, match="checkpoint"):
        PrecomputedDenseRetriever("different-model", root)
