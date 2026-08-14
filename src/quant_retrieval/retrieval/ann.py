"""Approximate nearest neighbour search over the corpus embeddings.

At 26,152 documents this is not worth using. Exact search over that corpus runs
at about 5ms, and every millisecond an approximate index could save is smaller
than the noise in the measurement. It exists to answer a different question: how
large does a corpus have to get before approximate search starts to pay. That is
a real result. "We added FAISS and it got faster" at this scale would not be.

HNSW builds a layered graph where each document links to its neighbours, and a
search walks the graph greedily instead of comparing against everything. The
knob is `ef_search`, the size of the candidate list kept while walking. Larger
means slower and closer to exact. Sweeping it is what produces the recall against
latency curve, so nothing here picks a value: the caller does.

Recall is measured against exact search on the same embeddings rather than
against the qrels. Two different things can go wrong in a retrieval system, the
model choosing badly and the index losing documents the model would have ranked
highly, and mixing them into one number tells you neither.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_retrieval.retrieval.base import SearchResult


class ApproximateRetriever:
    """Cosine search over prebuilt embeddings, through an HNSW graph.

    Takes embeddings rather than a model. Building an approximate index is a
    question about the index, and re-encoding the corpus for every value of
    `ef_search` would waste minutes measuring something that does not change.
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        *,
        neighbours: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        exact: bool = False,
    ) -> None:
        if neighbours < 1:
            raise ValueError("neighbours must be at least 1")
        if ef_search < 1:
            raise ValueError("ef_search must be at least 1")
        self.embeddings_path = Path(embeddings_path)
        self.neighbours = neighbours
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # The same class builds the exact index, so the comparison runs through
        # one code path and cannot differ by accident.
        self.exact = exact
        self.document_ids = np.array([], dtype=np.int64)
        self._index = None

    def index(self, document_ids: list[int], texts: list[str]) -> None:  # noqa: ARG002
        """Load the prebuilt embeddings and build the graph.

        `texts` is ignored and only present because the harness passes it. The
        embeddings must already exist, written by scripts/export_index.py, and
        their order must match `document_ids`.
        """
        import faiss

        embeddings = np.load(self.embeddings_path).astype(np.float32, copy=False)
        if len(embeddings) != len(document_ids):
            raise ValueError(
                f"{self.embeddings_path} holds {len(embeddings)} vectors "
                f"but the corpus has {len(document_ids)} documents"
            )
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

        dimensions = embeddings.shape[1]
        if self.exact:
            self._index = faiss.IndexFlatIP(dimensions)
        else:
            self._index = faiss.IndexHNSWFlat(
                dimensions, self.neighbours, faiss.METRIC_INNER_PRODUCT
            )
            self._index.hnsw.efConstruction = self.ef_construction
            self._index.hnsw.efSearch = self.ef_search
        self._index.add(embeddings)
        self.document_ids = np.asarray(document_ids, dtype=np.int64)

    def search_vector(self, query: np.ndarray, k: int) -> list[SearchResult]:
        """Search with an already encoded query."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._index is None:
            raise RuntimeError("index must be called before search")

        import faiss

        vector = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32, copy=False))
        faiss.normalize_L2(vector)
        scores, positions = self._index.search(vector, min(k, len(self.document_ids)))

        results = []
        for score, position in zip(scores[0], positions[0], strict=True):
            # HNSW returns -1 when it finds fewer than k, which happens at small
            # ef_search. Those are misses, not documents.
            if position < 0:
                continue
            results.append(
                SearchResult(document_id=int(self.document_ids[position]), score=float(score))
            )
        return results

    def search(self, query: str, k: int) -> list[SearchResult]:
        raise NotImplementedError(
            "ApproximateRetriever scores prebuilt vectors and cannot encode text. "
            "Encode the query first and call search_vector."
        )


def recall_against_exact(
    approximate: list[SearchResult], exact: list[SearchResult], k: int
) -> float:
    """Share of the exact top k that the approximate index also found.

    This is the quantity the whole study turns on. A fast index that quietly
    drops a third of the right answers is not a faster system, it is a worse one.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    wanted = {result.document_id for result in exact[:k]}
    if not wanted:
        return 0.0
    found = {result.document_id for result in approximate[:k]}
    return len(wanted & found) / len(wanted)
