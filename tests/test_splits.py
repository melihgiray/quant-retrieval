import pandas as pd
import pytest

from quant_retrieval.data.splits import (
    assign_time_splits,
    drop_cross_split_duplicates,
    duplicate_edges,
    normalize_title,
)


def make_queries(n=100, titles=None):
    return pd.DataFrame(
        {
            "question_id": range(1, n + 1),
            "creation_date": pd.date_range("2011-01-01", periods=n, freq="D"),
            "title": titles or [f"question {i}" for i in range(1, n + 1)],
        }
    )


def make_links(rows):
    return pd.DataFrame(rows, columns=["post_id", "related_post_id", "link_type_id"])


def test_splits_are_sized_as_asked():
    split = assign_time_splits(make_queries(100), val_frac=0.1, test_frac=0.1)
    assert split["split"].value_counts().to_dict() == {"train": 80, "val": 10, "test": 10}


def test_test_split_holds_the_newest_questions():
    split = assign_time_splits(make_queries(100))
    newest = split[split["split"] == "test"]["creation_date"].min()
    oldest_of_train = split[split["split"] == "train"]["creation_date"].max()
    assert newest > oldest_of_train


def test_splits_do_not_overlap_in_time():
    split = assign_time_splits(make_queries(100))
    bounds = split.groupby("split")["creation_date"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"]
    assert bounds.loc["val", "max"] < bounds.loc["test", "min"]


def test_impossible_fractions_are_rejected():
    with pytest.raises(ValueError):
        assign_time_splits(make_queries(10), val_frac=0.6, test_frac=0.6)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pricing a Swaption?", "pricing a swaption"),
        ("  What is  VaR? ", "what is var"),
        ("Black-Scholes vs. Bachelier", "black scholes vs bachelier"),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def test_duplicate_edges_come_from_links_and_from_repeated_titles():
    queries = make_queries(3, titles=["What is VaR?", "something else", "what is var"])
    links = make_links([(1, 2, 3), (1, 2, 1)])  # type 3 is a duplicate, type 1 is not
    edges = duplicate_edges(queries, links)
    assert (1, 2) in edges  # from the duplicate link
    assert (1, 3) in edges  # from the matching title
    assert len(edges) == 2  # the plain "linked" edge is ignored


def test_a_train_question_duplicating_a_test_question_is_dropped():
    queries = assign_time_splits(make_queries(100))
    train_id = int(queries[queries["split"] == "train"]["question_id"].iloc[0])
    test_id = int(queries[queries["split"] == "test"]["question_id"].iloc[0])
    links = make_links([(train_id, test_id, 3)])

    kept, dropped = drop_cross_split_duplicates(queries, links)

    assert train_id not in set(kept["question_id"])
    assert test_id in set(kept["question_id"])
    assert list(dropped["question_id"]) == [train_id]


def test_duplicates_inside_one_split_are_left_alone():
    queries = assign_time_splits(make_queries(100))
    train_ids = queries[queries["split"] == "train"]["question_id"].iloc[:2].tolist()
    links = make_links([(train_ids[0], train_ids[1], 3)])

    kept, dropped = drop_cross_split_duplicates(queries, links)

    assert dropped.empty
    assert len(kept) == len(queries)


def test_a_chain_of_duplicates_resolves_to_the_strongest_split():
    # train duplicates val, val duplicates test. All three are the same question,
    # so only the test copy may survive.
    queries = assign_time_splits(make_queries(100))
    train_id = int(queries[queries["split"] == "train"]["question_id"].iloc[0])
    val_id = int(queries[queries["split"] == "val"]["question_id"].iloc[0])
    test_id = int(queries[queries["split"] == "test"]["question_id"].iloc[0])
    links = make_links([(train_id, val_id, 3), (val_id, test_id, 3)])

    kept, dropped = drop_cross_split_duplicates(queries, links)

    assert set(dropped["question_id"]) == {train_id, val_id}
    assert test_id in set(kept["question_id"])


def test_no_test_question_survives_in_training_after_dedup():
    # The guarantee the whole split policy exists to provide.
    queries = assign_time_splits(make_queries(200))
    test_ids = queries[queries["split"] == "test"]["question_id"].tolist()
    links = make_links([(i, test_ids[0], 3) for i in queries["question_id"].iloc[:5]])

    kept, _ = drop_cross_split_duplicates(queries, links)
    train_ids = set(kept[kept["split"] == "train"]["question_id"])

    assert train_ids.isdisjoint(set(test_ids))
    assert train_ids.isdisjoint({i for i in queries["question_id"].iloc[:5]})
