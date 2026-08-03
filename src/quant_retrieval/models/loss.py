"""Contrastive objectives used to train the retrieval encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def info_nce_loss(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Treat the aligned document as positive and every other document as negative."""
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("query and document embeddings must be two-dimensional")
    if query_embeddings.shape != document_embeddings.shape:
        raise ValueError("query and document embeddings must have the same shape")
    if query_embeddings.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two pairs for in-batch negatives")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    queries = functional.normalize(query_embeddings, p=2, dim=1)
    documents = functional.normalize(document_embeddings, p=2, dim=1)
    logits = queries @ documents.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return functional.cross_entropy(logits, labels)


def in_batch_accuracy(query_embeddings: Tensor, document_embeddings: Tensor) -> float:
    """Share of queries whose own document is the closest one in the batch.

    A readable companion to the loss while training. Loss in nats says little on
    its own, but "62 percent of questions rank their own answer top of a batch of
    64" is a number you can reason about. It is not a retrieval metric: the
    competition is 63 random documents, not the full corpus.
    """
    queries = functional.normalize(query_embeddings, p=2, dim=1)
    documents = functional.normalize(document_embeddings, p=2, dim=1)
    predicted = (queries @ documents.transpose(0, 1)).argmax(dim=1)
    labels = torch.arange(predicted.shape[0], device=predicted.device)
    return (predicted == labels).float().mean().item()
