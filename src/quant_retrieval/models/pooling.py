"""Pooling shared by the trained encoder and the retrieval index.

Training and search have to turn token vectors into a sentence vector the exact
same way. If they drift apart, the model is optimised for one representation and
queried with another, the metrics still compute, and nothing anywhere says the
two halves disagree. So there is one implementation and both sides import it.
"""

from __future__ import annotations

from torch import Tensor

POOLING_STRATEGIES = ("mean", "cls")


def mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Average non-padding token vectors for each sequence in a batch."""
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    token_sum = (token_embeddings * expanded_mask).sum(dim=1)
    token_count = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return token_sum / token_count


def cls_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:  # noqa: ARG001
    """Take the first token's vector.

    Standard for BERT classification heads and, for this base model, expected to
    be worse: all-MiniLM-L6-v2 was distilled with mean pooling, so the first
    token was never trained to summarise the sequence. Kept so that claim is
    something the results show rather than something the README asserts.
    """
    return token_embeddings[:, 0]


def pool(strategy: str, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    if strategy == "mean":
        return mean_pool(token_embeddings, attention_mask)
    if strategy == "cls":
        return cls_pool(token_embeddings, attention_mask)
    raise ValueError(f"unknown pooling {strategy!r}, expected one of {POOLING_STRATEGIES}")
