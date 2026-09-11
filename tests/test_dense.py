from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from quant_retrieval.retrieval.dense import DenseRetriever, choose_device, mean_pool


def test_mean_pool_ignores_padding_tokens():
    embeddings = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[2.0, 4.0], [100.0, 100.0], [100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert torch.equal(mean_pool(embeddings, mask), torch.tensor([[2.0, 4.0], [2.0, 4.0]]))


def test_explicit_device_is_not_overridden():
    assert choose_device("cpu") == "cpu"


def test_dense_search_ranks_cosine_similarity():
    retriever = DenseRetriever("unused", show_progress=False)
    retriever.document_ids = np.array([30, 10, 20])
    retriever.embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32
    )

    def fake_encode(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    retriever._encode = MethodType(fake_encode, retriever)
    results = retriever.search("query", 2)
    assert [result.document_id for result in results] == [30, 20]
    assert results[0].score == pytest.approx(1.0)


def test_dense_search_breaks_ties_by_document_id():
    retriever = DenseRetriever("unused", show_progress=False)
    retriever.document_ids = np.array([20, 10])
    retriever.embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    retriever._encode = MethodType(
        lambda self, texts: np.array([[1.0, 0.0]], dtype=np.float32), retriever
    )
    assert [result.document_id for result in retriever.search("query", 2)] == [10, 20]


def test_dense_retriever_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="batch_size"):
        DenseRetriever("unused", batch_size=0)
    retriever = DenseRetriever("unused")
    with pytest.raises(ValueError, match="empty"):
        retriever.index([], [])
    with pytest.raises(RuntimeError, match="index"):
        retriever.search("query", 10)


def test_dense_retriever_loads_a_memory_mapped_index(tmp_path: Path):
    ids_path = tmp_path / "answer_ids.npy"
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(ids_path, np.array([30, 10], dtype=np.int64))
    np.save(embeddings_path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16))
    retriever = DenseRetriever("unused", show_progress=False)

    retriever.load_index(ids_path, embeddings_path)

    assert isinstance(retriever.document_ids, np.memmap)
    assert isinstance(retriever.embeddings, np.memmap)
    assert retriever.embeddings.dtype == np.float16


def test_dense_retriever_rejects_misaligned_precomputed_index(tmp_path: Path):
    ids_path = tmp_path / "answer_ids.npy"
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(ids_path, np.array([1, 2], dtype=np.int64))
    np.save(embeddings_path, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="same length"):
        DenseRetriever("unused").load_index(ids_path, embeddings_path)
