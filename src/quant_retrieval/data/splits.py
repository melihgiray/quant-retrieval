"""Split queries by time, then remove near duplicates that straddle a split.

Splitting by date rather than at random is the point. A random split lets the
model train on a question asked the same week as a test question, often by the
same person about the same paper, and every number comes out flattering. Sorting
by date and holding out the most recent slice asks the question that matters:
does this work on questions nobody had asked yet.

Time alone is not enough. Stack Exchange is full of the same question asked
years apart, and the site records those as duplicate links. A duplicate pair
split across train and test is the same leak wearing a different hat, so those
questions are dropped from the earlier split. Identical titles are treated as
duplicate edges too, since not every repeat gets flagged by a moderator.
"""

from __future__ import annotations

import re

import pandas as pd

SPLITS = ("train", "val", "test")
# Test wins over validation, validation wins over train. When a duplicate group
# spans splits, the members in the weaker split are the ones that go.
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
LINK_TYPE_DUPLICATE = 3

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. For exact repeat detection."""
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def assign_time_splits(
    queries: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1
) -> pd.DataFrame:
    """Add a `split` column. Oldest questions train, newest test."""
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be between 0 and 1")

    ordered = queries.sort_values(["creation_date", "question_id"]).reset_index(drop=True)
    total = len(ordered)
    n_test = round(total * test_frac)
    n_val = round(total * val_frac)
    n_train = total - n_val - n_test

    ordered["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return ordered


def split_boundaries(queries: pd.DataFrame) -> dict[str, dict[str, str]]:
    """The first and last question date in each split, for the dataset write-up."""
    return {
        split: {
            "first": str(group["creation_date"].min()),
            "last": str(group["creation_date"].max()),
            "queries": int(len(group)),
        }
        for split, group in queries.groupby("split")
    }


def duplicate_edges(queries: pd.DataFrame, post_links: pd.DataFrame) -> list[tuple[int, int]]:
    """Pairs of questions that are the same question, from links and from titles."""
    known = set(queries["question_id"])

    flagged = post_links[post_links["link_type_id"] == LINK_TYPE_DUPLICATE]
    edges = [
        (int(a), int(b))
        for a, b in zip(flagged["post_id"], flagged["related_post_id"], strict=True)
        if a in known and b in known
    ]

    by_title: dict[str, list[int]] = {}
    for question_id, title in zip(queries["question_id"], queries["title"], strict=True):
        key = normalize_title(title)
        if key:
            by_title.setdefault(key, []).append(int(question_id))
    for repeats in by_title.values():
        first = repeats[0]
        edges.extend((first, other) for other in repeats[1:])

    return edges


def drop_cross_split_duplicates(
    queries: pd.DataFrame, post_links: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove questions that duplicate a question in a higher priority split.

    Returns the surviving queries and a table of what was dropped and why.
    """
    groups = _UnionFind()
    for left, right in duplicate_edges(queries, post_links):
        groups.union(left, right)

    split_of = dict(zip(queries["question_id"], queries["split"], strict=True))
    best_priority: dict[int, int] = {}
    for question_id, split in split_of.items():
        if question_id in groups.parent:
            root = groups.find(question_id)
            priority = SPLIT_PRIORITY[split]
            best_priority[root] = max(best_priority.get(root, -1), priority)

    dropped_rows = []
    for question_id, split in split_of.items():
        if question_id not in groups.parent:
            continue
        winner = best_priority[groups.find(question_id)]
        if SPLIT_PRIORITY[split] < winner:
            dropped_rows.append(
                {
                    "question_id": question_id,
                    "split": split,
                    "reason": "duplicate of a question in " + _split_name(winner),
                }
            )

    dropped = pd.DataFrame(dropped_rows, columns=["question_id", "split", "reason"])
    kept = queries[~queries["question_id"].isin(set(dropped["question_id"]))].reset_index(drop=True)
    return kept, dropped


def _split_name(priority: int) -> str:
    return next(name for name, value in SPLIT_PRIORITY.items() if value == priority)
