import math

import pytest

from quant_retrieval.eval.metrics import (
    aggregate_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_counts_unique_relevant_documents():
    assert recall_at_k([8, 8, 4, 2], {2, 4}, 3) == 0.5


def test_recall_with_no_relevance_is_zero():
    assert recall_at_k([1, 2], set(), 10) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank_at_k([9, 4, 2, 1], {1, 2}, 10) == 1 / 3


def test_reciprocal_rank_respects_cutoff():
    assert reciprocal_rank_at_k([9, 4, 2], {2}, 2) == 0.0


def test_ndcg_matches_a_hand_calculated_example():
    # Grades [2, 0, 1] produce 3 + 1/log2(4) = 3.5 DCG.
    # The ideal [2, 1, 0] produces 3 + 1/log2(3).
    expected = 3.5 / (3 + 1 / math.log2(3))
    assert ndcg_at_k([10, 99, 20], {10: 2, 20: 1}, 3) == pytest.approx(expected)


def test_ndcg_is_one_for_an_ideal_ranking():
    assert ndcg_at_k([10, 20, 30], {10: 2, 20: 1}, 10) == 1.0


def test_aggregate_reports_strict_and_graded_views():
    rankings = {1: [12, 11], 2: [99, 21]}
    qrels = {1: {11: 2, 12: 1}, 2: {21: 2}}

    metrics = aggregate_metrics(rankings, qrels)

    assert metrics["mrr_at_10"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["recall_at_100"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    assert metrics["graded_ndcg_at_10"] > metrics["ndcg_at_10"]


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank_at_k, ndcg_at_k])
def test_metrics_reject_non_positive_cutoffs(metric):
    with pytest.raises(ValueError, match="positive"):
        metric([], {}, 0)


def test_aggregate_rejects_empty_qrels():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_metrics({}, {})
