"""Offline consistency checks for deployment assets, without loading the encoder."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.checkpoint import verify_checkpoint
from quant_retrieval.retrieval.index_artifacts import (
    corpus_ids,
    read_manifest,
    validate_document_ids,
    verify_export_files,
)
from quant_retrieval.retrieval.vectors import validate_embeddings
from quant_retrieval.serve.artifacts import CHECKSUM_FILE, file_sha256, verify_snapshot


def validate_snapshot_corpus(path: Path, manifest: dict) -> pd.DataFrame:
    for field in ("documents", "dimensions", "max_length"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError(f"snapshot {field} must be a positive integer")
    if manifest.get("pooling", "mean") != "mean":
        raise ValueError("snapshot serving requires mean pooling")
    corpus = pd.read_parquet(path)
    corpus_ids(corpus)
    if "question_id" not in corpus:
        raise ValueError("snapshot corpus requires question_id")
    questions = corpus["question_id"]
    if (not pd.api.types.is_integer_dtype(questions.dtype) or questions.isna().any()
            or (questions <= 0).any() or (questions > np.iinfo(np.int64).max).any()):
        raise ValueError("snapshot question IDs must be positive int64 integers")
    if len(corpus) != manifest["documents"]:
        raise ValueError("snapshot corpus document count differs from manifest")
    expected = manifest.get("corpus_sha256")
    if expected is not None and expected != corpus_fingerprint(corpus):
        raise ValueError("snapshot corpus fingerprint mismatch")
    return corpus


def validate_snapshot_vectors(directory: Path, manifest: dict, corpus: pd.DataFrame) -> None:
    verify_export_files(directory, manifest, ("answer_ids.npy", "embeddings_fp16.npy"))
    ids = np.load(directory / "answer_ids.npy", allow_pickle=False, mmap_mode="r")
    validate_document_ids(ids)
    if not np.array_equal(ids, corpus_ids(corpus)):
        raise ValueError("snapshot document IDs must match corpus order")
    vectors = np.load(directory / "embeddings_fp16.npy", allow_pickle=False, mmap_mode="r")
    if vectors.dtype != np.float16:
        raise ValueError("snapshot embeddings must use float16")
    validate_embeddings(vectors, len(ids))
    if vectors.shape[1] != manifest["dimensions"]:
        raise ValueError("snapshot vector dimensions differ from manifest")


def inspect_snapshot(root: Path) -> dict:
    """Report static consistency, not runtime readiness or retrieval quality."""
    files = verify_snapshot(root)
    manifest = read_manifest(root / "demo/manifest.json")
    checkpoint_verified = verify_checkpoint(root, manifest)
    corpus = validate_snapshot_corpus(root / "demo/corpus.parquet", manifest)
    validate_snapshot_vectors(root / "demo", manifest, corpus)
    return {
        "schema_version": 1,
        "scope": "offline_snapshot_consistency",
        "runtime_tested": False,
        "files": list(files),
        "checksums_sha256": file_sha256(root / CHECKSUM_FILE),
        "documents": manifest["documents"],
        "dimensions": manifest["dimensions"],
        "checkpoint_identity_verified": checkpoint_verified,
        "corpus_identity_verified": "corpus_sha256" in manifest,
        "export_identity_verified": "sha256" in manifest,
    }
