import pandas as pd
import pytest

from quant_retrieval.eval.sampling import sample_queries


def test_sampling_is_seeded_and_independent_of_input_order():
    queries = pd.DataFrame({"query_id": range(20), "split": ["val"] * 19 + ["test"]})
    selected = sample_queries(queries, 7, 17)
    pd.testing.assert_frame_equal(selected, sample_queries(queries.iloc[::-1], 7, 17))
    assert len(selected) == 7
    assert 19 not in selected.query_id.tolist()
    assert selected.query_id.tolist() != sample_queries(queries, 7, 18).query_id.tolist()


def test_sampling_caps_count_and_rejects_empty_or_held_out_split():
    queries = pd.DataFrame({"query_id": [1], "split": ["val"]})
    assert len(sample_queries(queries, 9, 17)) == 1
    for count, split in [(0, "val"), (1, "train"), (1, "test")]:
        with pytest.raises(ValueError):
            sample_queries(queries, count, 17, split)


def test_sampling_rejects_duplicate_identity():
    queries = pd.DataFrame({"query_id": [1, 1], "split": ["val", "val"]})
    with pytest.raises(ValueError, match="unique"):
        sample_queries(queries, 1, 17)
