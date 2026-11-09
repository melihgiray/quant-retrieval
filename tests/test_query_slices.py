import pytest

from quant_retrieval.eval.slices import grouped_summary
from tests.test_result_audit import result_record


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
