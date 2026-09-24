import pytest

from quant_retrieval.eval.diagnostics import ranking_diagnostics


def test_diagnostics_distinguish_retrieval_and_ordering_failures():
    rankings = {1: [9, 2, 3], 2: [9, 8, 3], 3: [9], 4: [9]}
    qrels = {1: {2: 2, 3: 1}, 2: {3: 2}, 3: {2: 2}, 4: {9: 1}}
    result = ranking_diagnostics(rankings, qrels, cutoff=2)
    assert result[1] == {
        "status": "top_k", "cutoff": 2, "first_primary_rank": 2,
        "primary_labels": 1, "primary_retrieved": 1, "sibling_retrieved": 1,
        "unjudged_retrieved": 1, "returned": 3,
    }
    assert result[2]["status"] == "below_cutoff"
    assert result[3]["status"] == "not_retrieved"
    assert result[3]["first_primary_rank"] is None
    assert result[4]["status"] == "no_primary_label"


def test_empty_ranking_is_a_miss_and_invalid_cutoff_fails():
    assert ranking_diagnostics({1: []}, {1: {2: 2}})[1]["status"] == "not_retrieved"
    with pytest.raises(ValueError, match="cutoff"):
        ranking_diagnostics({}, {}, 0)
