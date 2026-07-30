"""Okapi BM25 over a sparse document term matrix."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from scipy import sparse

from quant_retrieval.retrieval.base import SearchResult

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase lexical tokens, retaining decimals and dotted names."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """BM25 with precomputed document weights in a CSR matrix."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        self.document_ids = np.array([], dtype=np.int64)
        self.vocabulary: dict[str, int] = {}
        self.weights = sparse.csr_matrix((0, 0), dtype=np.float32)

    def index(self, document_ids: list[int], texts: list[str]) -> None:
        if len(document_ids) != len(texts):
            raise ValueError("document_ids and texts must have the same length")
        if not document_ids:
            raise ValueError("cannot index an empty corpus")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")

        token_counts = [Counter(tokenize(text)) for text in texts]
        terms = sorted({term for counts in token_counts for term in counts})
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        self.document_ids = np.asarray(document_ids, dtype=np.int64)

        document_lengths = np.asarray([sum(counts.values()) for counts in token_counts])
        average_length = float(document_lengths.mean())
        average_length = average_length or 1.0

        document_frequency = Counter(
            term for counts in token_counts for term in counts
        )
        corpus_size = len(texts)
        idf = {
            term: math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row, counts in enumerate(token_counts):
            length_normalizer = 1 - self.b + self.b * document_lengths[row] / average_length
            for term, frequency in counts.items():
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_normalizer
                rows.append(row)
                columns.append(self.vocabulary[term])
                values.append(idf[term] * numerator / denominator)

        self.weights = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, columns)),
            shape=(corpus_size, len(self.vocabulary)),
        )

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if self.weights.shape[0] == 0:
            raise RuntimeError("index must be called before search")

        term_ids = sorted(
            {self.vocabulary[token] for token in tokenize(query) if token in self.vocabulary}
        )
        if not term_ids:
            return []

        scores = np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()
        matched = np.flatnonzero(scores > 0)
        if not len(matched):
            return []

        # Sorting all matched rows is predictable and still cheap for this 26k
        # document corpus. Document ID is the stable tie breaker.
        order = np.lexsort((self.document_ids[matched], -scores[matched]))
        rows = matched[order[:k]]
        return [
            SearchResult(document_id=int(self.document_ids[row]), score=float(scores[row]))
            for row in rows
        ]
