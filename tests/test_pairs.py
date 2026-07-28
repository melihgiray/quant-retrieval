import pandas as pd
import pytest

from quant_retrieval.data.pairs import PairConfig, build_corpus, build_qrels


def make_questions(rows):
    return pd.DataFrame(rows, columns=["question_id", "accepted_answer_id"])


def make_answers(rows):
    return pd.DataFrame(rows, columns=["answer_id", "question_id", "score", "body_html"])


@pytest.fixture
def qrels():
    questions = make_questions(
        [
            (1, 30),  # accepted answer
            (2, None),  # no accepted, clear top scorer
            (3, None),  # no accepted, tied at the top
            (4, None),  # no accepted, top scorer too weak
        ]
    )
    answers = make_answers(
        [
            (30, 1, 8, "<p>accepted</p>"),
            (31, 1, 3, "<p>runner up</p>"),
            (32, 1, -2, "<p>downvoted</p>"),
            (40, 2, 5, "<p>clear winner</p>"),
            (41, 2, 1, "<p>second</p>"),
            (50, 3, 4, "<p>tied</p>"),
            (51, 3, 4, "<p>also tied</p>"),
            (60, 4, 1, "<p>weak best</p>"),
        ]
    )
    return build_qrels(questions, answers)


def grade_of(qrels, question_id, answer_id):
    row = qrels[(qrels["question_id"] == question_id) & (qrels["answer_id"] == answer_id)]
    return None if row.empty else int(row.iloc[0]["grade"])


def test_accepted_answer_is_the_primary_judgement(qrels):
    assert grade_of(qrels, 1, 30) == 2
    assert qrels[qrels["answer_id"] == 30].iloc[0]["label_source"] == "accepted"


def test_other_answers_on_a_judged_question_are_partially_relevant(qrels):
    assert grade_of(qrels, 1, 31) == 1


def test_downvoted_answers_are_not_relevant_at_all(qrels):
    assert grade_of(qrels, 1, 32) is None


def test_clear_top_scorer_stands_in_for_a_missing_accept(qrels):
    assert grade_of(qrels, 2, 40) == 2
    assert qrels[qrels["answer_id"] == 40].iloc[0]["label_source"] == "top_voted"


def test_a_tie_at_the_top_is_not_promoted(qrels):
    # Nobody can say which of two equally voted answers the asker wanted.
    assert qrels[qrels["question_id"] == 3].empty


def test_a_weak_top_scorer_is_not_promoted(qrels):
    assert qrels[qrels["question_id"] == 4].empty


def test_min_top_score_is_configurable():
    questions = make_questions([(4, None)])
    answers = make_answers([(60, 4, 1, "<p>weak best</p>")])
    relaxed = build_qrels(questions, answers, PairConfig(min_top_score=1))
    assert grade_of(relaxed, 4, 60) == 2


def test_every_question_has_at_most_one_primary_judgement(qrels):
    primary = qrels[qrels["grade"] == 2]
    assert primary["question_id"].is_unique


def test_corpus_keeps_short_answers_but_drops_empty_ones():
    answers = make_answers(
        [
            (1, 1, 0, "<p>ok</p>"),
            (2, 1, 0, ""),
            (3, 1, 0, "<img src='x.png'>"),  # nothing left once the image goes
        ]
    )
    corpus = build_corpus(answers)
    assert list(corpus["answer_id"]) == [1]
