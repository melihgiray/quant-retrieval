"""Validation shared by encoded and exported cosine indexes."""

import numpy as np


def validate_embeddings(
    vectors: np.ndarray, rows: int, *, atol: float = 1e-2, chunk_size: int = 4096
) -> None:
    if type(chunk_size) is not int or chunk_size <= 0 or not np.isfinite(atol) or atol < 0:
        raise ValueError("chunk size must be positive and tolerance finite and nonnegative")
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if vectors.shape[0] != rows or rows < 1:
        raise ValueError("embeddings and document IDs must have the same nonzero length")
    if vectors.shape[1] == 0:
        raise ValueError("embeddings must have at least one dimension")
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError("embeddings must be floating point")
    for start in range(0, rows, chunk_size):
        block = vectors[start:start + chunk_size]
        if not np.isfinite(block).all():
            raise ValueError("embeddings must be finite")
        if np.any(np.abs(block) > 1 + atol):
            raise ValueError("embeddings must be unit normalized")
        norms = np.linalg.norm(block.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=atol, rtol=0):
            raise ValueError("embeddings must be unit normalized")
