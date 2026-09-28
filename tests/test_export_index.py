import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scripts import export_index

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.index_artifacts import (
    corpus_ids,
    publish_index,
    verify_export_files,
)


@pytest.mark.parametrize("ids", [[0], [-1], [True], [1.5], [None], [1, 1], [2**63]])
def test_export_rejects_lossy_or_ambiguous_document_ids(ids):
    corpus = pd.DataFrame({"answer_id": ids, "text": ["answer"] * len(ids)})
    with pytest.raises(ValueError, match="IDs"):
        corpus_ids(corpus)


@pytest.mark.parametrize("text", [None, 4, " "])
def test_export_rejects_unusable_document_text(text):
    with pytest.raises(ValueError, match="text"):
        corpus_ids(pd.DataFrame({"answer_id": [1], "text": [text]}))


def test_export_preserves_document_order_and_int64_identity():
    corpus = pd.DataFrame({"answer_id": [3, 1, 2], "text": ["a", "b", "c"]})
    ids = corpus_ids(corpus)
    assert ids.dtype == np.int64
    assert ids.tolist() == [3, 1, 2]


def prepare_export(tmp_path, monkeypatch, embeddings):
    pd.DataFrame({"answer_id": [3, 1, 2], "text": ["first", "second", "third"]
                  }).to_parquet(tmp_path / "corpus.parquet")
    monkeypatch.setattr(export_index, "DenseRetriever", lambda *args, **kwargs:
                        SimpleNamespace(_encode=lambda texts: embeddings, device="cpu"))
    output = tmp_path / "export"
    monkeypatch.setattr("sys.argv", ["export", "--data", str(tmp_path), "--out", str(output)])
    return output


@pytest.mark.parametrize("embeddings", [np.ones((2, 3)), np.zeros((3, 3)),
    np.full((3, 3), float("nan")), np.eye(3, dtype=np.int32)])
def test_bad_encoder_output_is_rejected_before_export_files(tmp_path, monkeypatch, embeddings):
    output = prepare_export(tmp_path, monkeypatch, embeddings)
    with pytest.raises(ValueError, match="embeddings"):
        export_index.main()
    assert not output.exists()


def test_export_cli_keeps_ids_aligned_across_both_precisions(tmp_path, monkeypatch):
    output = prepare_export(tmp_path, monkeypatch, np.eye(3, dtype=np.float32))
    export_index.main()
    assert np.load(output / "answer_ids.npy").tolist() == [3, 1, 2]
    for name, dtype in [("fp32", np.float32), ("fp16", np.float16)]:
        vectors = np.load(output / f"embeddings_{name}.npy")
        assert vectors.dtype == dtype
        np.testing.assert_array_equal(vectors, np.eye(3))


def test_failed_export_does_not_publish_partial_files(tmp_path, monkeypatch):
    original = np.save

    def fail_half(path, array):
        if path.name == "embeddings_fp16.npy":
            raise OSError("disk full")
        original(path, array)

    monkeypatch.setattr(np, "save", fail_half)
    with pytest.raises(OSError, match="disk full"):
        publish_index(tmp_path / "new", np.array([1, 2]), np.eye(2), {})
    assert list(tmp_path.iterdir()) == []


def test_existing_exports_cannot_be_replaced(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("previous export")
    with pytest.raises(FileExistsError):
        publish_index(destination, np.array([1]), np.ones((1, 1)), {})
    assert marker.read_text() == "previous export"


def test_export_records_the_exact_corpus_and_encoding_settings(tmp_path, monkeypatch):
    output = prepare_export(tmp_path, monkeypatch, np.eye(3, dtype=np.float32))
    export_index.main()
    manifest = json.loads((output / "manifest.json").read_text())
    corpus = pd.read_parquet(tmp_path / "corpus.parquet")
    assert manifest["corpus_sha256"] == corpus_fingerprint(corpus)
    assert manifest["corpus_sha256"] != corpus_fingerprint(corpus.iloc[::-1])
    assert (manifest["seed"], manifest["batch_size"], manifest["pooling"]) == (17, 128, "mean")


def test_export_hashes_detect_changed_vectors_even_with_the_same_shape(tmp_path, monkeypatch):
    output = prepare_export(tmp_path, monkeypatch, np.eye(3, dtype=np.float32))
    export_index.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert verify_export_files(output, manifest)
    np.save(output / "embeddings_fp32.npy", np.eye(3, dtype=np.float32)[::-1])
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_export_files(output, manifest)


def test_legacy_exports_are_distinguished_from_incomplete_versioned_exports(tmp_path):
    assert verify_export_files(tmp_path, {}) is False
    with pytest.raises(ValueError, match="missing payload checksums"):
        verify_export_files(tmp_path, {"schema_version": 1})
    with pytest.raises(ValueError, match="all exported payloads"):
        verify_export_files(tmp_path, {"sha256": {"../outside": "0" * 64}})
