import torch
from torch import nn

from quant_retrieval.models.encoder import parameter_groups


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
