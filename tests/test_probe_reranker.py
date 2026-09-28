import pandas as pd
import pytest
import torch
from scripts.probe_reranker import eligible_distractors, probe_split


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
