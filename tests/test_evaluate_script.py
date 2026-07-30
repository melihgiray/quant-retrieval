import pytest
import torch
from scripts.evaluate import build_retriever, set_seed

from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever


def test_builds_bm25_from_config_parameters():
    retriever = build_retriever(
        {"retriever": "bm25", "parameters": {"k1": 1.6, "b": 0.4}}
    )
    assert isinstance(retriever, BM25Retriever)
    assert retriever.k1 == 1.6
    assert retriever.b == 0.4


def test_rejects_unknown_retrievers():
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever({"retriever": "magic"})


def test_builds_dense_retriever_without_loading_a_model():
    retriever = build_retriever(
        {
            "retriever": "dense",
            "parameters": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "cpu",
            },
        }
    )
    assert isinstance(retriever, DenseRetriever)
    assert retriever.batch_size == 32
    assert retriever.device == "cpu"
    assert retriever._model is None


def test_seed_repeats_torch_random_values():
    set_seed(17)
    first = torch.rand(4)
    set_seed(17)
    assert torch.equal(first, torch.rand(4))
