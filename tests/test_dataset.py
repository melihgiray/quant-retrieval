import pandas as pd
import pytest
import torch

from quant_retrieval.models.dataset import PairCollator, TrainingPair, load_training_pairs
from quant_retrieval.models.train import TrainingConfig, build_dataloader


class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer. One token per word, padded."""

    def __call__(self, texts, padding=True, truncation=True, max_length=8, return_tensors="pt"):
        sequences = [
            [hash(word) % 100 + 1 for word in text.split()][:max_length] for text in texts
        ]
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.tensor(
            [sequence + [0] * (width - len(sequence)) for sequence in sequences]
        )
        attention_mask = torch.tensor(
            [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def tables():
    corpus = pd.DataFrame(
        {
            "answer_id": [10, 11, 12, 13],
            "question_id": [1, 1, 2, 3],
            "score": [5, 1, 4, 2],
            "text": ["accepted one", "sibling one", "accepted two", "accepted three"],
        }
    )
    queries = pd.DataFrame(
        {
            "question_id": [1, 2, 3],
            "text": ["question one", "question two", "question three"],
            "split": ["train", "train", "val"],
        }
    )
    qrels = pd.DataFrame(
        {
            "question_id": [1, 1, 2, 3],
            "answer_id": [10, 11, 12, 13],
            "grade": [2, 1, 2, 2],
            "label_source": ["accepted", "sibling", "accepted", "accepted"],
        }
    )
    return corpus, queries, qrels


def test_pairs_come_from_the_requested_split_only(tables):
    pairs = load_training_pairs(*tables, split="train")
    assert sorted(pair.question_id for pair in pairs) == [1, 2]


def test_sibling_judgements_are_not_training_targets(tables):
    # Answer 11 is grade 1. Training on it would teach the model that one
    # question maps to several answers, which the in-batch loss cannot express.
    pairs = load_training_pairs(*tables, split="train")
    assert all(pair.document_text != "sibling one" for pair in pairs)


def test_pair_text_is_the_query_and_its_primary_answer(tables):
    pairs = {pair.question_id: pair for pair in load_training_pairs(*tables, split="train")}
    assert pairs[1].query_text == "question one"
    assert pairs[1].document_text == "accepted one"


def test_a_judgement_whose_answer_left_the_corpus_is_dropped(tables):
    corpus, queries, qrels = tables
    corpus = corpus[corpus["answer_id"] != 10]
    pairs = load_training_pairs(corpus, queries, qrels, split="train")
    assert [pair.question_id for pair in pairs] == [2]


def test_an_empty_split_is_an_error(tables):
    with pytest.raises(ValueError, match="no queries"):
        load_training_pairs(*tables, split="test")


def test_collator_returns_both_sides_with_the_same_batch_size():
    pairs = [TrainingPair(1, "a question here", "an answer"), TrainingPair(2, "another", "text")]
    batch = PairCollator(FakeTokenizer(), max_length=8)(pairs)

    assert set(batch) == {
        "query_input_ids",
        "query_attention_mask",
        "document_input_ids",
        "document_attention_mask",
    }
    assert batch["query_input_ids"].shape[0] == 2
    assert batch["document_input_ids"].shape[0] == 2
    assert batch["query_input_ids"].shape == batch["query_attention_mask"].shape


def test_collator_masks_padding():
    pairs = [TrainingPair(1, "one two three", "x"), TrainingPair(2, "one", "y")]
    batch = PairCollator(FakeTokenizer(), max_length=8)(pairs)
    assert batch["query_attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_collator_rejects_a_nonpositive_max_length():
    with pytest.raises(ValueError, match="max_length"):
        PairCollator(FakeTokenizer(), max_length=0)


def test_dataloader_drops_a_partial_final_batch(tmp_path):
    # Five pairs at batch size 2 gives two full batches. The odd one is dropped,
    # because a batch of one has no in-batch negatives and no defined loss.
    pairs = [TrainingPair(i, f"query {i}", f"document {i}") for i in range(5)]
    config = TrainingConfig(output_dir=tmp_path, batch_size=2)
    loader = build_dataloader(pairs, config, FakeTokenizer())

    batches = list(loader)
    assert len(batches) == 2
    assert all(batch["query_input_ids"].shape[0] == 2 for batch in batches)


def test_dataloader_refuses_fewer_pairs_than_one_batch(tmp_path):
    config = TrainingConfig(output_dir=tmp_path, batch_size=8)
    with pytest.raises(ValueError, match="fewer than one batch"):
        build_dataloader([TrainingPair(1, "q", "d")], config, FakeTokenizer())


def test_shuffling_is_reproducible_for_a_seed(tmp_path):
    pairs = [TrainingPair(i, f"query {i}", f"document {i}") for i in range(16)]
    config = TrainingConfig(output_dir=tmp_path, batch_size=4, seed=17)

    first = [
        b["query_input_ids"].tolist() for b in build_dataloader(pairs, config, FakeTokenizer())
    ]
    second = [
        b["query_input_ids"].tolist() for b in build_dataloader(pairs, config, FakeTokenizer())
    ]
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 1}, "batch_size"),
        ({"epochs": 0}, "epochs"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
    ],
)
def test_training_config_rejects_impossible_settings(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(output_dir=tmp_path, **kwargs)


@pytest.fixture
def mined():
    return pd.DataFrame(
        {
            "question_id": [1, 1, 1, 2],
            "answer_id": [12, 13, 11, 10],
            "rank": [3, 1, 2, 1],
            "source": ["bm25", "bm25", "dense", "bm25"],
        }
    )


def test_negatives_are_attached_hardest_first(tables, mined):
    corpus, queries, qrels = tables
    corpus = pd.concat(
        [
            corpus,
            pd.DataFrame(
                {
                    "answer_id": [14],
                    "question_id": [4],
                    "score": [0],
                    "text": ["unrelated"],
                }
            ),
        ]
    )
    pairs = load_training_pairs(
        corpus, queries, qrels, split="train", negatives=mined, negatives_per_query=2
    )
    first = next(pair for pair in pairs if pair.question_id == 1)
    # Ranks 1 and 2 come before rank 3.
    assert first.negative_texts == ("accepted three", "sibling one")


def test_a_question_without_enough_negatives_is_dropped(tables, mined):
    pairs = load_training_pairs(
        *tables, split="train", negatives=mined, negatives_per_query=3
    )
    # Question 2 has only one mined negative, so it cannot fill a batch slot.
    assert [pair.question_id for pair in pairs] == [1]


def test_no_negatives_requested_means_none_attached(tables, mined):
    pairs = load_training_pairs(*tables, split="train", negatives=mined)
    assert all(pair.negative_texts == () for pair in pairs)


def test_requesting_negatives_without_supplying_them_is_an_error(tables):
    with pytest.raises(ValueError, match="no negatives"):
        load_training_pairs(*tables, split="train", negatives_per_query=2)


def test_collator_flattens_negatives_across_the_batch():
    pairs = [
        TrainingPair(1, "query one", "document one", ("bad a", "bad b")),
        TrainingPair(2, "query two", "document two", ("bad c", "bad d")),
    ]
    batch = PairCollator(FakeTokenizer(), max_length=8)(pairs)
    # Two pairs with two negatives each gives four rows.
    assert batch["negative_input_ids"].shape[0] == 4
    assert batch["query_input_ids"].shape[0] == 2


def test_collator_rejects_a_batch_with_uneven_negative_counts():
    pairs = [
        TrainingPair(1, "query one", "document one", ("bad a",)),
        TrainingPair(2, "query two", "document two", ("bad b", "bad c")),
    ]
    with pytest.raises(ValueError, match="same number of negatives"):
        PairCollator(FakeTokenizer(), max_length=8)(pairs)


def test_config_rejects_negatives_without_a_path(tmp_path):
    with pytest.raises(ValueError, match="negatives_path"):
        TrainingConfig(output_dir=tmp_path, negatives_per_query=4)
