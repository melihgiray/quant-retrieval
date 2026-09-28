"""Stable top-k selection, including ties crossing the cutoff."""

import numpy as np


def top_k_rows(scores: np.ndarray, document_ids: np.ndarray, k: int) -> np.ndarray:
    limit = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - limit)[len(scores) - limit]
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((document_ids[candidates], -scores[candidates]))
    return candidates[order[:limit]]
