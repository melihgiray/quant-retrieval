import json
from copy import deepcopy

import numpy as np
import pytest
from scripts import inspect_queries

from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES
from quant_retrieval.retrieval.index_artifacts import file_digest
from quant_retrieval.retrieval.query_artifacts import (
    load_queries,
    publish_queries,
    validate_query_manifest,
    verify_query_compatibility,
)


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


def test_loaded_queries_are_read_only_and_detect_same_shape_changes(tmp_path):
    root = tmp_path / "queries"
    publish_queries(root, np.eye(2, dtype=np.float32), query_metadata())
    metadata, vectors = load_queries(root)
    assert metadata["question_ids"] == [20, 10]
    assert isinstance(vectors, np.memmap) and not vectors.flags.writeable
    np.save(root / "query_vectors.npy", np.eye(2, dtype=np.float32)[::-1])
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_queries(root)


def test_checksums_do_not_replace_query_vector_validation(tmp_path):
    root = tmp_path / "queries"
    manifest = publish_queries(root, np.eye(2, dtype=np.float32), query_metadata())
    np.save(root / "query_vectors.npy", np.zeros((2, 2), dtype=np.float32))
    manifest["vectors_sha256"] = file_digest(root / "query_vectors.npy")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normalized"):
        load_queries(root)


def test_query_loader_rejects_linked_payloads(tmp_path):
    root = tmp_path / "queries"
    publish_queries(root, np.eye(2, dtype=np.float32), query_metadata())
    path = root / "query_vectors.npy"
    path.rename(tmp_path / "outside.npy")
    path.symlink_to(tmp_path / "outside.npy")
    with pytest.raises(ValueError, match="unlinked"):
        load_queries(root)


def test_cached_queries_match_encoder_contents_not_paths_or_corpus_size():
    queries = query_metadata()
    index = {**queries, "checkpoint": "/another/machine/model", "documents": 100000}
    verify_query_compatibility(queries, index)
    for key, value in [("dimensions", 3), ("max_length", 128), ("pooling", "cls"),
                       ("checkpoint_sha256", None),
                       ("checkpoint_sha256", {name: "d" * 64 for name in CHECKPOINT_FILES})]:
        with pytest.raises(ValueError):
            verify_query_compatibility(queries, {**index, key: value})


def test_query_inspection_reports_storage_and_preserves_inputs(tmp_path, monkeypatch):
    root, output = tmp_path / "queries", tmp_path / "inspection.json"
    publish_queries(root, np.eye(2, dtype=np.float32), query_metadata())
    monkeypatch.setattr("sys.argv", ["inspect", "--queries", str(root), "--output", str(output)])
    inspect_queries.main()
    report = json.loads(output.read_text())
    assert report["array_bytes"] == 16
    assert report["file_bytes"] > report["array_bytes"]
    assert report["payload_checksum_verified"]
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        inspect_queries.main()
    assert saved == output.read_bytes()
    monkeypatch.setattr("sys.argv", ["inspect", "--queries", str(root),
                                    "--output", str(root / "manifest.json")])
    with pytest.raises(SystemExit):
        inspect_queries.main()
    load_queries(root)
