"""Stable query samples for validation-only performance studies."""

import pandas as pd


def sample_queries(queries: pd.DataFrame, count: int, seed: int, split: str = "val"):
    if split not in {"train", "val"}:
        raise ValueError("performance studies accept only train or val queries")
    if count <= 0:
        raise ValueError("query count must be positive")
    selected = queries.loc[queries["split"] == split].sort_values("query_id")
    if selected.empty:
        raise ValueError(f"no queries available for {split}")
    if selected["query_id"].duplicated().any():
        raise ValueError("query IDs must be unique within the selected split")
    return selected.sample(n=min(count, len(selected)), random_state=seed).reset_index(drop=True)
