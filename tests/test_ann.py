import os
import sys

import numpy as np
import pytest

from quant_retrieval.retrieval.ann import ApproximateRetriever, recall_against_exact
from quant_retrieval.retrieval.base import SearchResult

# faiss and torch each bundle an OpenMP runtime, and loading both in one process
# aborts on macOS with OMP Error #15. The documented workaround, setting
# KMP_DUPLICATE_LIB_OK, is described by its own authors as able to silently
# produce incorrect results, which is not a trade worth making in a repository
# whose point is trustworthy numbers.
#
# So these skip on macOS during the full suite, where torch is already loaded,
# and run everywhere else including the Colab machine that does the real ANN
# work. To run them here, use a process that never imports torch:
#
#     QR_FORCE_ANN_TESTS=1 pytest tests/test_ann.py
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("QR_FORCE_ANN_TESTS") != "1",
    reason="faiss and torch cannot share a process on macOS, see the note above",
)


@pytest.fixture
def corpus(tmp_path):
    """400 random unit vectors saved where a retriever can load them."""
    generator = np.random.default_rng(0)
    embeddings = generator.random((400, 32)).astype(np.float32)
    path = tmp_path / "embeddings.npy"
    np.save(path, embeddings)
    # Document ids deliberately not 0..399, so a positional bug cannot pass.
    document_ids = [1000 + i * 3 for i in range(400)]
    return path, document_ids, embeddings


def build(path, document_ids, **kwargs):
    retriever = ApproximateRetriever(path, **kwargs)
    retriever.index(document_ids, ["unused"] * len(document_ids))
    return retriever


def test_exact_search_finds_a_document_by_its_own_vector(corpus):
    path, document_ids, embeddings = corpus
    retriever = build(path, document_ids, exact=True)
    results = retriever.search_vector(embeddings[7], 3)
    assert results[0].document_id == document_ids[7]


def test_results_come_back_in_descending_score_order(corpus):
    path, document_ids, embeddings = corpus
    retriever = build(path, document_ids, exact=True)
    scores = [result.score for result in retriever.search_vector(embeddings[3], 10)]
    assert scores == sorted(scores, reverse=True)


def test_document_ids_are_mapped_not_positions(corpus):
    path, document_ids, embeddings = corpus
    retriever = build(path, document_ids, exact=True)
    returned = {result.document_id for result in retriever.search_vector(embeddings[0], 20)}
    assert returned <= set(document_ids)
    assert not returned & {0, 1, 2}


def test_a_higher_ef_search_gets_closer_to_exact(corpus):
    path, document_ids, embeddings = corpus
    exact = build(path, document_ids, exact=True)
    sloppy = build(path, document_ids, neighbours=4, ef_construction=8, ef_search=1)
    careful = build(path, document_ids, neighbours=4, ef_construction=8, ef_search=200)

    queries = embeddings[:40]
    def mean_recall(retriever):
        return np.mean([
            recall_against_exact(
                retriever.search_vector(query, 10), exact.search_vector(query, 10), 10
            )
            for query in queries
        ])

    assert mean_recall(careful) >= mean_recall(sloppy)
    assert mean_recall(careful) > 0.8


def test_asking_for_more_than_the_corpus_returns_the_corpus(corpus):
    path, document_ids, embeddings = corpus
    retriever = build(path, document_ids, exact=True)
    assert len(retriever.search_vector(embeddings[0], 10_000)) == len(document_ids)


def test_embeddings_that_do_not_match_the_corpus_are_rejected(corpus):
    path, document_ids, _ = corpus
    retriever = ApproximateRetriever(path)
    with pytest.raises(ValueError, match="holds 400 vectors"):
        retriever.index(document_ids[:10], ["unused"] * 10)


def test_searching_before_indexing_is_an_error(corpus):
    path, _, embeddings = corpus
    with pytest.raises(RuntimeError, match="index must be called"):
        ApproximateRetriever(path).search_vector(embeddings[0], 5)


def test_searching_with_text_says_what_to_do_instead(corpus):
    path, document_ids, _ = corpus
    retriever = build(path, document_ids, exact=True)
    with pytest.raises(NotImplementedError, match="search_vector"):
        retriever.search("a question", 5)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"neighbours": 0}, "neighbours"), ({"ef_search": 0}, "ef_search")]
)
def test_invalid_settings_are_rejected(corpus, kwargs, message):
    path, _, _ = corpus
    with pytest.raises(ValueError, match=message):
        ApproximateRetriever(path, **kwargs)


def test_k_must_be_positive(corpus):
    path, document_ids, embeddings = corpus
    retriever = build(path, document_ids, exact=True)
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search_vector(embeddings[0], 0)


def results(ids):
    return [SearchResult(document_id=i, score=1.0) for i in ids]


def test_recall_against_exact_counts_the_overlap():
    assert recall_against_exact(results([1, 2, 3]), results([1, 2, 3]), 3) == 1.0
    assert recall_against_exact(results([1, 9, 8]), results([1, 2, 3]), 3) == pytest.approx(1 / 3)
    assert recall_against_exact(results([7, 8, 9]), results([1, 2, 3]), 3) == 0.0


def test_recall_ignores_order_within_the_cutoff():
    # An approximate index that finds the same documents in a different order
    # has lost nothing. Ranking quality is what nDCG is for.
    assert recall_against_exact(results([3, 1, 2]), results([1, 2, 3]), 3) == 1.0


def test_recall_of_an_empty_exact_list_is_zero():
    assert recall_against_exact(results([1]), [], 3) == 0.0
