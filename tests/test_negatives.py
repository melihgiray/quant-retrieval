import pandas as pd
import pytest

from quant_retrieval.models.negatives import (
    RANDOM_RANK_OFFSET,
    NegativeMiningConfig,
    combine_negatives,
    mine_negatives,
    sample_random_negatives,
)
from quant_retrieval.retrieval.base import SearchResult


class ScriptedRetriever:
    """Returns a fixed ranking, so the mining rules are what is under test."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.indexed = None

    def index(self, document_ids, texts):
        self.indexed = list(document_ids)

    def search(self, query, k):
        return [SearchResult(document_id=i, score=1.0 / (rank + 1)) for rank, i in
                enumerate(self.ranking)][:k]


@pytest.fixture
def tables():
    # Question 1 owns answers 10 (accepted), 11 (sibling) and 12 (downvoted,
    # so unjudged). Everything from 20 up belongs to other questions.
    corpus = pd.DataFrame(
        {
            "answer_id": [10, 11, 12, 20, 21, 22, 23],
            "question_id": [1, 1, 1, 2, 3, 4, 5],
            "score": [5, 1, -3, 2, 2, 2, 2],
            "text": [f"answer {i}" for i in [10, 11, 12, 20, 21, 22, 23]],
        }
    )
    queries = pd.DataFrame({"question_id": [1], "text": ["question one"], "split": ["train"]})
    qrels = pd.DataFrame(
        {
            "question_id": [1, 1],
            "answer_id": [10, 11],
            "grade": [2, 1],
            "label_source": ["accepted", "sibling"],
        }
    )
    return corpus, queries, qrels


def mine(tables, ranking, **kwargs):
    corpus, queries, qrels = tables
    config = NegativeMiningConfig(**{"depth": 50, "per_query": 4, "skip_top": 0, **kwargs})
    return mine_negatives(
        ScriptedRetriever(ranking), corpus, queries, qrels, config=config,
        source="test", show_progress=False,
    )


def test_the_judged_positive_is_never_a_negative(tables):
    negatives = mine(tables, [10, 20, 21, 22, 23])
    assert 10 not in set(negatives["answer_id"])


def test_a_grade_one_sibling_is_never_a_negative(tables):
    # Answer 11 is a real answer to this question. Pushing it away teaches
    # the model something false.
    negatives = mine(tables, [11, 20, 21, 22, 23])
    assert 11 not in set(negatives["answer_id"])


def test_an_unjudged_answer_to_the_same_question_is_never_a_negative(tables):
    # Answer 12 was downvoted so it carries no judgement, but it is still on
    # topic and cannot be called wrong.
    negatives = mine(tables, [12, 20, 21, 22, 23])
    assert 12 not in set(negatives["answer_id"])


def test_negatives_come_from_other_questions_in_rank_order(tables):
    negatives = mine(tables, [20, 21, 22, 23])
    assert list(negatives["answer_id"]) == [20, 21, 22, 23]
    assert list(negatives["rank"]) == [0, 1, 2, 3]


def test_only_per_query_negatives_are_kept(tables):
    negatives = mine(tables, [20, 21, 22, 23], per_query=2)
    assert list(negatives["answer_id"]) == [20, 21]


def test_skip_top_drops_the_highest_ranked_hits(tables):
    negatives = mine(tables, [20, 21, 22, 23], skip_top=2, per_query=2)
    assert list(negatives["answer_id"]) == [22, 23]


def test_excluded_answers_do_not_use_up_the_quota(tables):
    # The three same-question answers are skipped, so the quota still fills.
    negatives = mine(tables, [10, 11, 12, 20, 21], per_query=2)
    assert list(negatives["answer_id"]) == [20, 21]


def test_mining_indexes_the_whole_corpus(tables):
    corpus, queries, qrels = tables
    retriever = ScriptedRetriever([20, 21])
    mine_negatives(retriever, corpus, queries, qrels, show_progress=False)
    assert retriever.indexed == [10, 11, 12, 20, 21, 22, 23]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"per_query": 0}, "per_query"), ({"depth": 1, "skip_top": 1}, "depth")],
)
def test_impossible_mining_settings_are_rejected(tables, kwargs, message):
    with pytest.raises(ValueError, match=message):
        mine(tables, [20, 21], **kwargs)


def test_combine_keeps_one_row_per_pair_at_its_best_rank():
    first = pd.DataFrame(
        {"question_id": [1, 1], "answer_id": [20, 21], "rank": [5, 1], "source": ["bm25"] * 2}
    )
    second = pd.DataFrame(
        {"question_id": [1], "answer_id": [20], "rank": [2], "source": ["dense"]}
    )
    combined = combine_negatives(first, second)

    assert len(combined) == 2
    row = combined[combined["answer_id"] == 20].iloc[0]
    assert row["rank"] == 2
    assert row["source"] == "dense"


def test_combine_rejects_an_empty_call():
    with pytest.raises(ValueError, match="nothing to combine"):
        combine_negatives()


def draw(tables, **kwargs):
    corpus, queries, qrels = tables
    return sample_random_negatives(
        corpus, queries, qrels, show_progress=False, **{"per_query": 2, "seed": 17, **kwargs}
    )


def test_random_negatives_never_include_the_judged_positive(tables):
    assert 10 not in set(draw(tables, per_query=3)["answer_id"])


def test_random_negatives_never_include_a_sibling(tables):
    assert 11 not in set(draw(tables, per_query=3)["answer_id"])


def test_random_negatives_never_include_an_unjudged_same_question_answer(tables):
    # Answer 12 is unjudged but belongs to question 1, so it is not wrong.
    assert 12 not in set(draw(tables, per_query=3)["answer_id"])


def test_random_negatives_return_the_requested_count(tables):
    assert list(draw(tables, per_query=3).groupby("question_id").size()) == [3]


def test_random_negatives_are_labelled_and_ranked_above_mined_ones(tables):
    drawn = draw(tables)
    assert set(drawn["source"]) == {"random"}
    assert drawn["rank"].min() >= RANDOM_RANK_OFFSET


def test_random_negatives_repeat_for_a_seed_and_differ_across_seeds(tables):
    first = draw(tables, seed=5)
    assert first.equals(draw(tables, seed=5))
    assert not first.equals(draw(tables, seed=6))


def test_combining_keeps_the_mined_rank_over_the_random_one(tables):
    mined = pd.DataFrame(
        {"question_id": [1], "answer_id": [20], "rank": [3], "source": ["bm25"]}
    )
    random_table = pd.DataFrame(
        {"question_id": [1], "answer_id": [20], "rank": [RANDOM_RANK_OFFSET], "source": ["random"]}
    )
    combined = combine_negatives(mined, random_table)
    assert len(combined) == 1
    assert combined.iloc[0]["source"] == "bm25"


def test_random_negatives_reject_a_nonpositive_count(tables):
    with pytest.raises(ValueError, match="per_query"):
        draw(tables, per_query=0)


def test_random_negatives_reject_a_corpus_too_small_to_draw_from(tables):
    corpus, queries, qrels = tables
    with pytest.raises(ValueError, match="too small"):
        sample_random_negatives(
            corpus.head(2), queries, qrels, per_query=5, show_progress=False
        )
