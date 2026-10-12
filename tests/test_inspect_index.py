import json
import sys

import numpy as np
import pytest
from scripts import inspect_index as inspection_script
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


@pytest.mark.parametrize("field", ["documents", "dimensions"])
@pytest.mark.parametrize("value", [True, 1.0, 0, None])
def test_inspection_requires_integer_manifest_dimensions(tmp_path, field, value):
    manifest = {"documents": 1, "dimensions": 1, field: value}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive integer"):
        inspect_index(tmp_path)


def test_inspection_checks_existing_output_before_reading_inputs(tmp_path, monkeypatch):
    output = tmp_path / "keep.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["inspect", "--artifacts", str(tmp_path / "missing"),
                                     "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        inspection_script.main()
    assert error.value.code == 2
    assert output.read_text() == "keep"
