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


def test_hybrid_specs_build_their_children():
    retriever = build_retriever(
        {
            "retriever": "hybrid",
            "parameters": {
                "rrf_k": 30,
                "retrievers": [
                    {"retriever": "bm25", "parameters": {"k1": 1.2}},
                    {"retriever": "bm25", "parameters": {"k1": 0.9}},
                ],
            },
        }
    )
    assert retriever.rrf_k == 30
    assert [r.k1 for r in retriever.retrievers] == [1.2, 0.9]


def test_reranking_specs_build_their_base():
    retriever = build_retriever(
        {
            "retriever": "rerank",
            "parameters": {
                "model_name": "checkpoints/nowhere",
                "depth": 25,
                "base": {"retriever": "bm25", "parameters": {"k1": 1.4}},
            },
        }
    )
    assert retriever.depth == 25
    assert retriever.base.k1 == 1.4


def test_a_pipeline_can_nest_more_than_one_level():
    retriever = build_retriever(
        {
            "retriever": "rerank",
            "parameters": {
                "model_name": "checkpoints/nowhere",
                "base": {
                    "retriever": "hybrid",
                    "parameters": {
                        "retrievers": [
                            {"retriever": "bm25", "parameters": {}},
                            {"retriever": "bm25", "parameters": {}},
                        ]
                    },
                },
            },
        }
    )
    assert len(retriever.base.retrievers) == 2


def test_building_a_spec_does_not_consume_it():
    # The spec is written into the committed result as provenance, so building
    # from it must not pop keys out of the caller's dictionary.
    spec = {
        "retriever": "hybrid",
        "parameters": {
            "retrievers": [
                {"retriever": "bm25", "parameters": {}},
                {"retriever": "bm25", "parameters": {}},
            ]
        },
    }
    build_retriever(spec)
    assert "retrievers" in spec["parameters"]
