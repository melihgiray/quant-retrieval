from types import MethodType

import numpy as np
import pytest

from quant_retrieval.retrieval.base import SearchResult
from quant_retrieval.retrieval.rerank import RerankingRetriever


class ScriptedRetriever:
    def __init__(self, ranking):
        self.ranking = ranking
        self.asked_for = None

    def index(self, document_ids, texts):
        self.indexed = list(document_ids)

    def search(self, query, k):
        self.asked_for = k
        return [
            SearchResult(document_id=i, score=1.0 / (rank + 1))
            for rank, i in enumerate(self.ranking)
        ][:k]


def build(ranking, scores_by_text, **kwargs):
    """A reranker whose cross-encoder is a lookup table instead of a model."""
    retriever = RerankingRetriever(ScriptedRetriever(ranking), "unused", **kwargs)
    retriever.documents = {i: f"doc{i}" for i in ranking}
    retriever._score = MethodType(
        lambda self, query, documents: np.array([scores_by_text[d] for d in documents]),
        retriever,
    )
    return retriever


def test_reranking_reorders_the_shortlist():
    # The base retriever ranks 1, 2, 3. The reranker likes 3 most.
    retriever = build([1, 2, 3], {"doc1": 0.1, "doc2": 0.2, "doc3": 0.9}, depth=3)
    assert [r.document_id for r in retriever.search("q", 3)] == [3, 2, 1]


def test_scores_come_from_the_reranker_not_the_base():
    retriever = build([1, 2], {"doc1": 0.25, "doc2": 0.75}, depth=2)
    assert retriever.search("q", 2)[0].score == pytest.approx(0.75)


def test_documents_past_the_depth_keep_their_order_below_the_reranked_ones():
    # Only the first two are rescored. Document 3 was never looked at, so it
    # stays where the base retriever put it, behind everything reranked.
    retriever = build([1, 2, 3], {"doc1": 0.1, "doc2": 0.9}, depth=2)
    assert [r.document_id for r in retriever.search("q", 3)] == [2, 1, 3]


def test_recall_past_the_depth_is_preserved():
    # A pipeline with depth 1 must still return 5 documents when asked for 5,
    # otherwise adding a reranker would silently destroy Recall@100.
    retriever = build([1, 2, 3, 4, 5], {"doc1": 0.5}, depth=1)
    assert [r.document_id for r in retriever.search("q", 5)] == [1, 2, 3, 4, 5]


def test_the_base_is_asked_for_at_least_the_depth():
    retriever = build([1, 2, 3], {f"doc{i}": 0.0 for i in (1, 2, 3)}, depth=3)
    retriever.search("q", 1)
    assert retriever.base.asked_for == 3


def test_the_base_is_asked_for_k_when_k_exceeds_depth():
    retriever = build([1, 2, 3], {"doc1": 0.0}, depth=1)
    retriever.search("q", 3)
    assert retriever.base.asked_for == 3


def test_ties_break_by_document_id():
    retriever = build([9, 4], {"doc9": 0.5, "doc4": 0.5}, depth=2)
    assert [r.document_id for r in retriever.search("q", 2)] == [4, 9]


def test_indexing_keeps_the_text_for_rescoring():
    retriever = RerankingRetriever(ScriptedRetriever([1]), "unused")
    retriever.index([1, 2], ["first", "second"])
    assert retriever.documents == {1: "first", 2: "second"}
    assert retriever.base.indexed == [1, 2]


def test_an_empty_shortlist_returns_nothing():
    retriever = build([], {}, depth=5)
    assert retriever.search("q", 5) == []


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"depth": 0}, "depth"), ({"batch_size": 0}, "batch_size")]
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RerankingRetriever(ScriptedRetriever([1]), "unused", **kwargs)


def test_k_must_be_positive():
    retriever = build([1], {"doc1": 1.0})
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("q", 0)
