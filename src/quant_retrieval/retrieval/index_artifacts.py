"""Input and storage contracts for reusable embedding exports."""

import numpy as np
import pandas as pd


def corpus_ids(corpus: pd.DataFrame) -> np.ndarray:
    if not {"answer_id", "text"} <= set(corpus.columns) or not corpus.columns.is_unique:
        raise ValueError("corpus needs unique answer_id and text columns")
    if corpus.empty:
        raise ValueError("cannot export an empty corpus")
    ids = corpus["answer_id"]
    if (not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any()
            or (ids <= 0).any() or (ids > np.iinfo(np.int64).max).any()):
        raise ValueError("corpus answer IDs must be positive int64 integers")
    if ids.duplicated().any():
        raise ValueError("corpus answer IDs must be unique")
    if not all(isinstance(text, str) and text.strip() for text in corpus["text"]):
        raise ValueError("corpus text must contain nonempty strings")
    return ids.to_numpy(dtype=np.int64)
