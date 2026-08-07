import pytest

from quant_retrieval.retrieval.base import SearchResult
from quant_retrieval.retrieval.hybrid import HybridRetriever


class ScriptedRetriever:
    def __init__(self, ranking):
        self.ranking = ranking
        self.indexed = None

    def index(self, document_ids, texts):
        self.indexed = list(document_ids)

    def search(self, query, k):
        return [
            SearchResult(document_id=i, score=1.0 / (rank + 1))
            for rank, i in enumerate(self.ranking)
        ][:k]


def fuse(first, second, **kwargs):
    return HybridRetriever([ScriptedRetriever(first), ScriptedRetriever(second)], **kwargs)


def test_a_document_both_retrievers_rank_highly_wins():
    # 2 is second for both. 1 and 3 are first for one and missing from the other.
    retriever = fuse([1, 2], [3, 2])
    assert [r.document_id for r in retriever.search("q", 1)] == [2]


def test_agreement_beats_a_single_first_place():
    retriever = fuse([1, 2, 3], [4, 2, 3], rrf_k=1)
    ranked = [r.document_id for r in retriever.search("q", 3)]
    assert ranked[0] == 2


def test_documents_only_one_retriever_found_still_appear():
    retriever = fuse([1], [2])
    assert {r.document_id for r in retriever.search("q", 2)} == {1, 2}


def test_a_small_rrf_k_sharpens_the_top_of_the_ranking():
    # With k=1 the first place contributes 1/2 and second 1/3, a wide gap. With
    # k=1000 they are nearly equal, so agreement further down matters more.
    sharp = fuse([1, 2, 3], [3, 2, 1], rrf_k=1).search("q", 3)
    flat = fuse([1, 2, 3], [3, 2, 1], rrf_k=1000).search("q", 3)
    assert sharp[0].score > flat[0].score


def test_weights_shift_the_balance():
    balanced = fuse([1], [2]).search("q", 2)
    assert balanced[0].score == pytest.approx(balanced[1].score)

    tilted = HybridRetriever(
        [ScriptedRetriever([1]), ScriptedRetriever([2])], weights=[3.0, 1.0]
    ).search("q", 2)
    assert tilted[0].document_id == 1
    assert tilted[0].score > tilted[1].score


def test_ties_break_by_document_id():
    retriever = HybridRetriever([ScriptedRetriever([9, 4]), ScriptedRetriever([9, 4])])
    assert [r.document_id for r in retriever.search("q", 2)] == [9, 4]

    swapped = HybridRetriever([ScriptedRetriever([4, 9]), ScriptedRetriever([9, 4])])
    # Both documents now score identically, so the smaller id comes first.
    assert [r.document_id for r in swapped.search("q", 2)] == [4, 9]


def test_indexing_reaches_every_retriever():
    first, second = ScriptedRetriever([1]), ScriptedRetriever([2])
    HybridRetriever([first, second]).index([1, 2], ["a", "b"])
    assert first.indexed == [1, 2]
    assert second.indexed == [1, 2]


def test_depth_reads_further_than_k_so_agreement_deep_in_a_list_still_counts():
    # Document 50 sits at rank 49 in the first list and rank 1 in the second.
    # Asking for 4 results reads at least `depth` from each, so with depth 50 it
    # picks up both contributions and with depth 2 only the second one.
    def fused_score(depth):
        retriever = HybridRetriever(
            [ScriptedRetriever(list(range(1, 51))), ScriptedRetriever([99, 50])],
            rrf_k=1,
            depth=depth,
        )
        return {r.document_id: r.score for r in retriever.search("q", 4)}[50]

    assert fused_score(50) > fused_score(2)


def test_depth_is_a_floor_not_a_cap():
    # A caller asking for more results than `depth` still gets them.
    retriever = HybridRetriever(
        [ScriptedRetriever([1, 2, 3]), ScriptedRetriever([4, 5, 6])], depth=1
    )
    assert len(retriever.search("q", 6)) == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rrf_k": 0}, "rrf_k"),
        ({"depth": 0}, "depth"),
        ({"weights": [1.0]}, "one entry per retriever"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HybridRetriever([ScriptedRetriever([1]), ScriptedRetriever([2])], **kwargs)


def test_fusion_needs_more_than_one_retriever():
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([ScriptedRetriever([1])])
