import pytest
import torch
from torch import nn

from quant_retrieval.models.encoder import parameter_groups
from quant_retrieval.models.pooling import cls_pool, pool


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)  # weight is 2d, bias is 1d
        self.norm = nn.LayerNorm(4)  # both parameters are 1d


def test_only_matrices_are_weight_decayed():
    decayed, undecayed = parameter_groups(TinyModel(), weight_decay=0.01)

    assert decayed["weight_decay"] == 0.01
    assert undecayed["weight_decay"] == 0.0
    # Just the linear weight matrix is decayed. The bias and both LayerNorm
    # parameters are not, because shrinking them toward zero distorts the
    # normalization for no benefit.
    assert [tuple(p.shape) for p in decayed["params"]] == [(4, 4)]
    assert len(undecayed["params"]) == 3


def test_frozen_parameters_are_left_out_entirely():
    model = TinyModel()
    model.linear.weight.requires_grad = False

    decayed, undecayed = parameter_groups(model, weight_decay=0.01)

    assert decayed["params"] == []
    assert len(undecayed["params"]) == 3


def test_every_trainable_parameter_lands_in_exactly_one_group():
    model = TinyModel()
    decayed, undecayed = parameter_groups(model, weight_decay=0.01)

    grouped = {id(p) for p in decayed["params"]} | {id(p) for p in undecayed["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped == trainable
    assert len(decayed["params"]) + len(undecayed["params"]) == len(trainable)


def test_groups_are_usable_by_an_optimizer():
    model = TinyModel()
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=1e-3)
    loss = model.norm(model.linear(torch.randn(2, 4))).sum()
    loss.backward()
    optimizer.step()


def test_cls_pooling_takes_the_first_token():
    embeddings = torch.tensor([[[1.0, 2.0], [9.0, 9.0]], [[3.0, 4.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1], [1, 0]])
    assert torch.equal(cls_pool(embeddings, mask), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_pool_dispatches_to_both_strategies():
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    mask = torch.tensor([[1, 1]])
    assert torch.equal(pool("mean", embeddings, mask), torch.tensor([[2.0, 4.0]]))
    assert torch.equal(pool("cls", embeddings, mask), torch.tensor([[1.0, 3.0]]))


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool("max", torch.ones(1, 2, 2), torch.ones(1, 2))
