import pytest

from quant_retrieval.eval.slices import count_buckets, diagnostic_labels, grouped_summary
from tests.test_result_audit import diagnostic_record, result_record


def test_groups_partition_queries_and_recover_the_overall_mean():
    result = grouped_summary(result_record(), {20: "long", 10: "short"}, group_by="fixture")
    groups = result["groups"]
    assert [row["label"] for row in groups] == ["long", "short"]
    assert sum(row["queries"] for row in groups) == result["queries"]
    assert sum(row["fraction"] * row["metrics"]["ndcg_at_10"] for row in groups) == 0.4
    assert all(row["small_group"] for row in groups)


@pytest.mark.parametrize(
    "labels", [{10: "a"}, {10: "a", 20: " "}, {"10": "a", "20": "b"}, {10: "a", 20: None}]
)
def test_groups_reject_missing_or_ambiguous_assignments(labels):
    with pytest.raises(ValueError):
        grouped_summary(result_record(), labels, group_by="fixture")


def test_small_group_threshold_marks_but_does_not_discard_evidence():
    result = grouped_summary(
        result_record(), {10: "all", 20: "all"}, group_by="fixture", min_queries=2
    )
    assert result["groups"][0]["small_group"] is False
    assert result["groups"][0]["question_ids"] == [10, 20]


def test_diagnostic_groups_keep_rank_and_label_count_axes_separate():
    record = diagnostic_record()
    assert diagnostic_labels(record, "diagnostic_status") == {10: "top_k", 20: "top_k"}
    assert diagnostic_labels(record, "primary_labels") == {10: "1", 20: "1"}
    assert diagnostic_labels(record, "first_primary_rank") == {10: "ranks_2_to_10", 20: "rank_1"}


def test_diagnostic_grouping_refuses_absent_evidence():
    with pytest.raises(ValueError, match="requires saved"):
        diagnostic_labels(result_record(), "diagnostic_status")
    with pytest.raises(ValueError, match="must be one of"):
        diagnostic_labels(diagnostic_record(), "unknown")


def test_count_buckets_cover_zero_and_exact_boundaries_without_overlap():
    assert count_buckets({1: 0, 2: 9, 3: 10, 4: 24, 5: 25}, (10, 25)) == {
        1: "0_to_9",
        2: "0_to_9",
        3: "10_to_24",
        4: "10_to_24",
        5: "25_plus",
    }


@pytest.mark.parametrize("edges", [(), (0,), (True,), (10, 10), (20, 10), (1.5,)])
def test_count_buckets_reject_ambiguous_boundaries(edges):
    with pytest.raises(ValueError, match="boundaries"):
        count_buckets({1: 5}, edges)


@pytest.mark.parametrize("counts", [{1: True}, {1: -1}, {1: 2.5}, {True: 2}, {}])
def test_count_buckets_reject_invalid_counts(counts):
    with pytest.raises(ValueError, match="bucket counts"):
        count_buckets(counts, (10,))
