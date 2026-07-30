import pytest
from scripts.evaluate import build_retriever

from quant_retrieval.retrieval.bm25 import BM25Retriever


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
