"""The trainable encoder behind both sides of the retrieval task.

One set of weights encodes questions and answers alike. A two tower model with
separate weights per side is the other option, and it is the wrong one here: the
corpus is 26k documents against 10k training pairs, so a second tower doubles the
parameters without doubling the supervision. A shared tower also means the served
system loads one model instead of two.

The forward pass ends in an L2 normalized vector, which is the same thing the
retrieval index stores. Cosine similarity is then a dot product, and what the
loss optimises is exactly what search compares.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn.functional as functional
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from quant_retrieval.models.pooling import POOLING_STRATEGIES, pool


class TextEncoder(nn.Module):
    """Hugging Face encoder body, pooled and normalized."""

    def __init__(self, model_name: str, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLING_STRATEGIES}")
        self.model_name = model_name
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool(self.pooling, output.last_hidden_state, attention_mask)
        return functional.normalize(pooled, p=2, dim=1)

    def save(self, directory: Path, tokenizer: AutoTokenizer | None = None) -> None:
        """Write a checkpoint the retriever can load by path.

        Saved in the Hugging Face layout on purpose. `DenseRetriever` takes a
        model name, and a directory path works anywhere a name does, so a tuned
        checkpoint is evaluated by the committed harness with no new code.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and undecayed groups.

    Biases and LayerNorm gains are excluded from weight decay. Decaying them
    pulls normalization statistics toward zero for no benefit, and every
    transformer fine-tuning recipe leaves them out.
    """
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
