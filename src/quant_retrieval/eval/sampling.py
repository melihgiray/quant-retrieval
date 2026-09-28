"""Stable query samples for validation-only performance studies."""

import pandas as pd


def sample_queries(queries: pd.DataFrame, count: int, seed: int, split: str = "val"):
    if split not in {"train", "val"}:
        raise ValueError("performance studies accept only train or val queries")
    if type(count) is not int or count <= 0:
        raise ValueError("query count must be positive")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between zero and 2**32 - 1")
    if not {"question_id", "text", "split"} <= set(queries.columns):
        raise ValueError("queries require question_id, text and split columns")
    if not queries.columns.is_unique:
        raise ValueError("query columns must be unique")
    selected = queries.loc[queries["split"] == split].sort_values("question_id")
    if selected.empty:
        raise ValueError(f"no queries available for {split}")
    ids = selected["question_id"]
    if not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any() or (ids <= 0).any():
        raise ValueError("query IDs must be positive integers")
    if not all(isinstance(text, str) and text.strip() for text in selected["text"]):
        raise ValueError("query texts must be nonempty strings")
    if selected["question_id"].duplicated().any():
        raise ValueError("query IDs must be unique within the selected split")
    return selected.sample(n=min(count, len(selected)), random_state=seed).reset_index(drop=True)
