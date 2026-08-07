"""Fuse several retrievers with reciprocal rank fusion.

BM25 and the tuned encoder fail differently. BM25 finds an exact ticker, a
function name or a formula that the encoder has smoothed into a general notion of
the topic. The encoder finds the answer that never repeats the question's words.
Running both and merging gets some of each.

Fusion is on rank, not score. BM25 scores are unbounded sums of term weights and
cosine similarities live in [-1, 1], so combining the numbers directly means
inventing a scale factor and then tuning it. Reciprocal rank fusion sidesteps
that: each retriever contributes 1/(k + rank) for the documents it ranked, and
the constant k decides how quickly the contribution decays. The usual k is 60,
which is large enough that the difference between rank 1 and rank 5 matters more
than the difference between rank 50 and rank 60.
"""

from __future__ import annotations

from collections.abc import Sequence

from quant_retrieval.retrieval.base import Retriever, SearchResult


class HybridRetriever:
    """Reciprocal rank fusion over two or more retrievers."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        depth: int = 100,
        weights: Sequence[float] | None = None,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("fusion needs at least two retrievers")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have one entry per retriever")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        # How deep to read from each retriever before fusing. A document one
        # retriever ranks 90th and the other misses entirely can still surface,
        # which is the point of fusing at all.
        self.depth = depth
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)

    def index(self, document_ids: list[int], texts: list[str]) -> None:
        for retriever in self.retrievers:
            retriever.index(document_ids, texts)

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        fused: dict[int, float] = {}
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for rank, result in enumerate(retriever.search(query, max(k, self.depth))):
                contribution = weight / (self.rrf_k + rank + 1)
                fused[result.document_id] = fused.get(result.document_id, 0.0) + contribution

        # Ties broken by document id, matching the other retrievers, so a run is
        # reproducible rather than dependent on dict ordering.
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(document_id=document_id, score=score)
            for document_id, score in ordered[:k]
        ]
