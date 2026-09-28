import numpy as np
import pytest

from quant_retrieval.retrieval.vectors import validate_embeddings


def test_vector_validation_bounds_temporary_allocations(tmp_path, monkeypatch):
    path = tmp_path / "vectors.npy"
    np.save(path, np.tile([1., 0.], (5, 1)).astype(np.float16))
    vectors = np.load(path, mmap_mode="r")
    original, sizes = np.linalg.norm, []

    def norm(block, **kwargs):
        sizes.append(len(block))
        return original(block, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", norm)
    validate_embeddings(vectors, 5, chunk_size=2)
    assert sizes == [2, 2, 1]
    assert isinstance(vectors, np.memmap)


def test_validation_checks_the_last_partial_chunk():
    vectors = np.tile([1., 0.], (5, 1))
    vectors[-1] = 0
    with pytest.raises(ValueError, match="unit normalized"):
        validate_embeddings(vectors, 5, chunk_size=2)
