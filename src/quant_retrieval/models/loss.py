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


def info_nce_with_negatives(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """InfoNCE where each query also competes against mined wrong answers.

    `negative_embeddings` is (batch, negatives_per_query, dimensions). The
    negatives are pooled across the whole batch rather than kept per query, so a
    batch of 64 with 4 negatives each puts 64 + 256 candidates in front of every
    question instead of 4. Sharing them costs nothing, since they are encoded
    either way, and every extra candidate is another chance for the model to be
    wrong in a way the loss can see.
    """
    if negative_embeddings.ndim != 3:
        raise ValueError("negative embeddings must be (batch, negatives, dimensions)")
    if negative_embeddings.shape[0] != query_embeddings.shape[0]:
        raise ValueError("negative embeddings must have one row per query")
    if query_embeddings.shape != document_embeddings.shape:
        raise ValueError("query and document embeddings must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size, _, dimensions = negative_embeddings.shape
    queries = functional.normalize(query_embeddings, p=2, dim=1)
    positives = functional.normalize(document_embeddings, p=2, dim=1)
    negatives = functional.normalize(negative_embeddings.reshape(-1, dimensions), p=2, dim=1)

    candidates = torch.cat([positives, negatives], dim=0)
    logits = queries @ candidates.transpose(0, 1) / temperature
    # Candidate i is query i's own answer, so the label is just the position.
    labels = torch.arange(batch_size, device=logits.device)
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
