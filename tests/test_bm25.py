import math

import pytest

from quant_retrieval.retrieval.bm25 import BM25Retriever, tokenize


def test_tokenizer_keeps_decimals_and_dotted_names():
    assert tokenize("S&P 500, 3.14 and numpy.linalg") == [
        "s",
        "p",
        "500",
        "3.14",
        "and",
        "numpy.linalg",
    ]


def test_bm25_ranks_a_repeated_match_first():
    retriever = BM25Retriever(k1=1.2, b=0.0)
    retriever.index([30, 10, 20], ["delta delta gamma", "delta", "gamma"])

    results = retriever.search("delta", k=3)

    assert [result.document_id for result in results] == [30, 10]

    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_first = idf * (2 * 2.2) / (2 + 1.2)
    assert results[0].score == pytest.approx(expected_first)
    assert results[1].score == pytest.approx(idf)


def test_length_normalization_prefers_the_shorter_document():
    retriever = BM25Retriever(b=0.75)
    retriever.index([1, 2], ["volatility", "volatility plus unrelated words"])

    assert [result.document_id for result in retriever.search("volatility", 2)] == [1, 2]


def test_search_uses_document_id_to_break_ties():
    retriever = BM25Retriever(b=0.0)
    retriever.index([20, 10], ["delta", "delta"])
    assert [result.document_id for result in retriever.search("delta", 2)] == [10, 20]


def test_unknown_terms_return_no_results():
    retriever = BM25Retriever()
    retriever.index([1], ["delta"])
    assert retriever.search("gamma", 10) == []


def test_index_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="empty"):
        BM25Retriever().index([], [])
    with pytest.raises(ValueError, match="same length"):
        BM25Retriever().index([1], [])
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever().index([1, 1], ["a", "b"])


def test_search_requires_an_index_and_positive_k():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError, match="index"):
        retriever.search("delta", 10)
    retriever.index([1], ["delta"])
    with pytest.raises(ValueError, match="positive"):
        retriever.search("delta", 0)
