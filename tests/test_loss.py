import math

import pytest
import torch

from quant_retrieval.models.loss import (
    in_batch_accuracy,
    info_nce_loss,
    info_nce_with_negatives,
)


def test_info_nce_matches_a_hand_calculated_example():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = queries.clone()

    loss = info_nce_loss(queries, documents, temperature=1.0)

    assert loss.item() == pytest.approx(math.log(math.e + 1) - 1)


def test_aligned_pairs_score_better_than_swapped_pairs():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = queries.clone()
    swapped = aligned.flip(0)

    assert info_nce_loss(queries, aligned) < info_nce_loss(queries, swapped)


def test_info_nce_backpropagates_into_both_sides():
    queries = torch.randn(4, 3, requires_grad=True)
    documents = torch.randn(4, 3, requires_grad=True)

    info_nce_loss(queries, documents).backward()

    assert queries.grad is not None
    assert documents.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(documents.grad).all()


@pytest.mark.parametrize(
    ("queries", "documents", "message"),
    [
        (torch.ones(2), torch.ones(2), "two-dimensional"),
        (torch.ones(2, 3), torch.ones(3, 3), "same shape"),
        (torch.ones(1, 3), torch.ones(1, 3), "at least two"),
    ],
)
def test_info_nce_rejects_invalid_batches(queries, documents, message):
    with pytest.raises(ValueError, match=message):
        info_nce_loss(queries, documents)


def test_info_nce_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        info_nce_loss(torch.ones(2, 3), torch.ones(2, 3), temperature=0)


def test_in_batch_accuracy_is_one_when_every_pair_matches_itself():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert in_batch_accuracy(queries, queries.clone()) == pytest.approx(1.0)


def test_in_batch_accuracy_is_zero_when_every_pair_is_swapped():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert in_batch_accuracy(queries, queries.flip(0)) == pytest.approx(0.0)


def test_in_batch_accuracy_counts_the_fraction_that_match():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    documents = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Only the first query ranks its own document first.
    assert in_batch_accuracy(queries, documents) == pytest.approx(1 / 3)


def test_in_batch_accuracy_ignores_vector_length():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert in_batch_accuracy(queries, queries * 7.0) == pytest.approx(1.0)


def test_hard_negatives_raise_the_loss_when_they_look_like_the_answer():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = queries.clone()
    far = torch.tensor([[[-1.0, 0.0]], [[0.0, -1.0]]])
    near = torch.tensor([[[0.95, 0.31]], [[0.31, 0.95]]])

    assert info_nce_with_negatives(queries, documents, near) > info_nce_with_negatives(
        queries, documents, far
    )


def test_negatives_are_shared_across_the_batch():
    # Query 0's own negative is harmless, but query 1's negative points straight
    # at query 0's answer. Sharing negatives means query 0 pays for it.
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = queries.clone()
    private = torch.tensor([[[-1.0, 0.0]], [[0.0, -1.0]]])
    shared = torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]])

    assert info_nce_with_negatives(queries, documents, shared) > info_nce_with_negatives(
        queries, documents, private
    )


def test_adding_negatives_never_lowers_the_loss():
    torch.manual_seed(0)
    queries = torch.randn(4, 8)
    documents = torch.randn(4, 8)
    negatives = torch.randn(4, 3, 8)

    with_negatives = info_nce_with_negatives(queries, documents, negatives)
    without = info_nce_loss(queries, documents)
    assert with_negatives >= without


def test_hard_negative_loss_backpropagates_into_the_negatives():
    queries = torch.randn(3, 4, requires_grad=True)
    documents = torch.randn(3, 4, requires_grad=True)
    negatives = torch.randn(3, 2, 4, requires_grad=True)

    info_nce_with_negatives(queries, documents, negatives).backward()

    assert negatives.grad is not None
    assert torch.isfinite(negatives.grad).all()


@pytest.mark.parametrize(
    ("negatives", "message"),
    [
        (torch.ones(2, 3), "batch, negatives, dimensions"),
        (torch.ones(3, 2, 3), "one row per query"),
    ],
)
def test_hard_negative_loss_rejects_bad_shapes(negatives, message):
    with pytest.raises(ValueError, match=message):
        info_nce_with_negatives(torch.ones(2, 3), torch.ones(2, 3), negatives)
