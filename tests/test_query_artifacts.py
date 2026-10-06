from copy import deepcopy

import numpy as np
import pytest

from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES
from quant_retrieval.retrieval.index_artifacts import file_digest
from quant_retrieval.retrieval.query_artifacts import publish_queries, validate_query_manifest


def query_metadata():
    return {"schema_version": 1, "kind": "ann_queries", "split": "val", "queries": 2,
            "dimensions": 2, "max_length": 64, "pooling": "mean", "dtype": "float32",
            "seed": 17, "question_ids": [20, 10], "query_sha256": "a" * 64,
            "vectors_sha256": "b" * 64,
            "checkpoint_sha256": {name: "c" * 64 for name in CHECKPOINT_FILES}}


def test_query_contract_keeps_sample_order_and_metadata_unchanged():
    metadata = query_metadata()
    before = deepcopy(metadata)
    validate_query_manifest(metadata)
    assert metadata == before


@pytest.mark.parametrize("key,value", [
    ("split", "test"), ("split", "train"), ("schema_version", True), ("kind", "documents"),
    ("question_ids", [20, 20]), ("question_ids", [2**63, 1]), ("question_ids", [True, 10]),
    ("queries", 3), ("dimensions", 0), ("seed", -1), ("dtype", "float16"),
    ("pooling", []), ("query_sha256", "unknown"), ("checkpoint_sha256", None),
])
def test_query_contract_rejects_ambiguous_or_incompatible_metadata(key, value):
    metadata = query_metadata()
    metadata[key] = value
    with pytest.raises(ValueError):
        validate_query_manifest(metadata)


def test_query_publication_preserves_order_and_refuses_replacement(tmp_path):
    destination = tmp_path / "queries"
    vectors = np.eye(2, dtype=np.float32)
    manifest = publish_queries(destination, vectors, query_metadata())
    assert manifest["question_ids"] == [20, 10]
    assert manifest["vectors_sha256"] == file_digest(destination / "query_vectors.npy")
    np.testing.assert_array_equal(np.load(destination / "query_vectors.npy"), vectors)
    with pytest.raises(FileExistsError):
        publish_queries(destination, vectors, query_metadata())


def test_failed_query_write_leaves_no_partial_version(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        publish_queries(tmp_path / "queries", np.eye(2, dtype=np.float32), query_metadata())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("vectors", [np.eye(2), np.zeros((2, 2), dtype=np.float32),
                                     np.eye(3, dtype=np.float32)])
def test_invalid_query_vectors_fail_before_publication(tmp_path, vectors):
    with pytest.raises(ValueError):
        publish_queries(tmp_path / "queries", vectors, query_metadata())
    assert list(tmp_path.iterdir()) == []
