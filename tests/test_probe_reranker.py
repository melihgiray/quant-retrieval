import pandas as pd
import pytest
import torch
from scripts.probe_reranker import eligible_distractors, probe_split, select_probe_queries


def dataset():
    corpus = pd.DataFrame({"answer_id": [1, 2, 3, 4, 5, 6],
        "question_id": [10, 10, 10, 20, 30, 30],
        "text": ["gold", "sibling", "unjudged sibling", "linked", "other a", "other b"]})
    queries = pd.DataFrame({"question_id": [10], "text": ["query"], "split": ["val"]})
    qrels = pd.DataFrame({"question_id": [10, 10, 10], "answer_id": [1, 2, 4], "grade": [2, 1, 1]})
    return corpus, queries, qrels


def run_probe(scores, distractors=2):
    captured = []

    def tokenizer(queries, candidates, **kwargs):
        captured.extend(candidates)
        return {"input_ids": torch.zeros((len(candidates), 1), dtype=torch.long)}

    report = probe_split(lambda **kwargs: torch.tensor(scores), tokenizer, *dataset(),
        split="val", questions=1, distractors=distractors, max_length=32, device="cpu", seed=17)
    return report, captured


def test_probe_excludes_all_judged_and_same_question_answers():
    corpus, _, qrels = dataset()
    assert eligible_distractors(corpus, qrels, 10).tolist() == [5, 6]
    report, captured = run_probe([3., 1., 0.])
    assert captured[0] == "gold"
    assert set(captured[1:]) == {"other a", "other b"}
    assert report["top_one_accuracy"] == 1


def test_probe_refuses_an_impossible_negative_pool():
    with pytest.raises(ValueError, match="eligible distractors"):
        run_probe([1., 0., 0., 0.], distractors=3)


@pytest.mark.parametrize("scores,accuracy,ties", [
    ([0., 0., 0.], 0, 1), ([1., 1., 0.], 0, 1),
    ([0., 1., 1.], 0, 0), ([2., 1., 0.], 1, 0),
])
def test_positive_position_does_not_win_score_ties(scores, accuracy, ties):
    report, _ = run_probe(scores)
    assert report["top_one_accuracy"] == accuracy
    assert report["tied_questions"] == ties
    assert report["tie_policy"] == "positive_must_strictly_outscore_all_distractors"


def test_probe_sampling_is_seeded_and_independent_of_row_order():
    queries = pd.DataFrame({"question_id": range(1, 21), "text": ["query"] * 20,
                            "split": ["val"] * 20})
    qrels = pd.DataFrame({"question_id": range(1, 21), "answer_id": range(101, 121),
                          "grade": [2] * 20})
    first, _ = select_probe_queries(queries, qrels, "val", 4, 17)
    reordered, _ = select_probe_queries(queries.iloc[::-1], qrels.iloc[::-1], "val", 4, 17)
    pd.testing.assert_frame_equal(first, reordered)
    assert first.question_id.tolist() != [1, 2, 3, 4]
    with pytest.raises(ValueError, match="no queries"):
        select_probe_queries(queries, qrels.iloc[:0], "val", 4, 17)
    with pytest.raises(ValueError, match="train or val"):
        select_probe_queries(queries, qrels, "test", 4, 17)


@pytest.mark.parametrize("scores", [[1.], [[1., 2., 3.]],
    [float("nan"), 0., 1.], [float("inf"), 0., 1.], [True, False, False]])
def test_probe_rejects_malformed_or_nonfinite_model_scores(scores):
    with pytest.raises(ValueError, match="one finite floating score"):
        run_probe(scores)


def test_probe_computes_large_finite_margins_without_float32_overflow():
    report, _ = run_probe([3e38, -3e38, -3e38])
    assert report["top_one_accuracy"] == 1
    assert report["median_margin"] == pytest.approx(6e38)
