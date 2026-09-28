"""Validation shared by encoded and exported cosine indexes."""

import numpy as np


def validate_embeddings(vectors: np.ndarray, rows: int, *, atol: float = 1e-2) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if vectors.shape[0] != rows or rows < 1:
        raise ValueError("embeddings and document IDs must have the same nonzero length")
    if vectors.shape[1] == 0:
        raise ValueError("embeddings must have at least one dimension")
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError("embeddings must be floating point")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=atol, rtol=0):
        raise ValueError("embeddings must be unit normalized")
