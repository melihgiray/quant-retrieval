"""Information retrieval metrics with no dependency on an evaluation library."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant documents found in the first ``k`` results."""
    _check_k(k)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(ranked_ids: Sequence[int], relevant_ids: set[int], k: int) -> float:
    """Reciprocal rank of the first relevant document, or zero when none is found."""
    _check_k(k)
    for rank, document_id in enumerate(ranked_ids[:k], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[int], relevance: Mapping[int, int], k: int) -> float:
    """Normalized discounted cumulative gain using exponential gain."""
    _check_k(k)
    if not relevance:
        return 0.0

    actual = _dcg([relevance.get(document_id, 0) for document_id in ranked_ids[:k]])
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = _dcg(ideal_grades)
    return actual / ideal if ideal else 0.0


def aggregate_metrics(
    rankings: Mapping[int, Sequence[int]],
    qrels: Mapping[int, Mapping[int, int]],
) -> dict[str, float]:
    """Average strict and graded metrics across the judged queries.

    Strict metrics count only grade 2 documents. Graded nDCG uses every
    positive grade and rewards a primary answer more than a sibling answer.
    """
    if not qrels:
        raise ValueError("qrels must contain at least one query")

    totals = {
        "ndcg_at_10": 0.0,
        "mrr_at_10": 0.0,
        "recall_at_10": 0.0,
        "recall_at_100": 0.0,
        "graded_ndcg_at_10": 0.0,
    }
    for query_id, grades in qrels.items():
        ranked = rankings.get(query_id, ())
        primary = {document_id for document_id, grade in grades.items() if grade >= 2}
        binary_primary = {document_id: 1 for document_id in primary}
        totals["ndcg_at_10"] += ndcg_at_k(ranked, binary_primary, 10)
        totals["mrr_at_10"] += reciprocal_rank_at_k(ranked, primary, 10)
        totals["recall_at_10"] += recall_at_k(ranked, primary, 10)
        totals["recall_at_100"] += recall_at_k(ranked, primary, 100)
        totals["graded_ndcg_at_10"] += ndcg_at_k(ranked, grades, 10)

    count = len(qrels)
    return {name: value / count for name, value in totals.items()}


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
