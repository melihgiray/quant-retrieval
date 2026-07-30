import math

import pytest
import torch

from quant_retrieval.models.loss import info_nce_loss


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
