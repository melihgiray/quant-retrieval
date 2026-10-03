import json

import numpy as np
import pandas as pd
import pytest

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES, checkpoint_hashes
from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.factory import build_retriever
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


def test_precision_configs_run_through_the_full_harness_without_document_encoding(
    tmp_path, monkeypatch
):
    corpus, root, checkpoint = fixture_export(tmp_path)
    queries = pd.DataFrame({"question_id": [10], "text": ["query"], "split": ["val"]})
    qrels = pd.DataFrame({"question_id": [10], "answer_id": [3], "grade": [2]})
    calls = []

    def encode(self, texts):
        calls.append(texts)
        assert texts == ["query"]
        return np.array([[1., 0., 0.]], dtype=np.float32)

    monkeypatch.setattr(DenseRetriever, "_encode", encode)
    reports = []
    for precision in ("fp16", "fp32"):
        retriever = build_retriever({"retriever": "precomputed_dense", "parameters": {
            "model_name": str(checkpoint), "artifacts_path": root,
            "precision": precision, "device": "cpu"}})
        reports.append(evaluate_retriever(retriever, corpus, queries, qrels))
    assert calls == [["query"], ["query"]]
    assert reports[0]["metrics"] == reports[1]["metrics"]
    assert reports[0]["metrics"]["mrr_at_10"] == 1


def test_verified_precomputed_model_can_move_but_cannot_change(tmp_path):
    corpus, root, original = fixture_export(tmp_path)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in CHECKPOINT_FILES:
        (relocated / name).write_text(name)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_sha256"] = checkpoint_hashes(relocated)
    path.write_text(json.dumps(manifest))
    retriever = PrecomputedDenseRetriever(str(relocated), root, device="cpu")
    assert retriever.checkpoint_verified
    assert not original.exists()
    retriever.index(corpus.answer_id.tolist(), corpus.text.tolist())
    previous = retriever.embeddings
    (relocated / "model.safetensors").write_text("different model")
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        retriever.index(corpus.answer_id.tolist(), corpus.text.tolist())
    assert retriever.embeddings is previous
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        PrecomputedDenseRetriever(str(relocated), root, device="cpu")
