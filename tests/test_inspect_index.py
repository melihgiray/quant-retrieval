import numpy as np
import pytest
from scripts.inspect_index import inspect_index

from quant_retrieval.retrieval.index_artifacts import publish_index


def test_precision_inspection_matches_direct_numeric_error(tmp_path):
    vectors = np.array([[.6, .8], [1., 0.], [.8, .6]], dtype=np.float32)
    root = tmp_path / "export"
    publish_index(root, np.array([1, 2, 3]), vectors, {"max_length": 32})
    report = inspect_index(root, chunk_size=2)
    delta = vectors.astype(np.float16).astype(np.float64) - vectors.astype(np.float64)
    assert report["payload_checksums_verified"] is True
    assert report["fp16_vs_fp32"]["max_absolute_error"] == pytest.approx(np.abs(delta).max())
    assert report["fp16_vs_fp32"]["root_mean_square_error"] == pytest.approx(
        np.sqrt(np.mean(delta**2)))
    assert report["storage"]["fp16"]["array_bytes"] * 2 == report["storage"]["fp32"]["array_bytes"]
