"""Input and storage contracts for reusable embedding exports."""

import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.eval.results import parse_record, write_result
from quant_retrieval.retrieval.vectors import validate_embeddings

INDEX_PAYLOADS = ("answer_ids.npy", "embeddings_fp32.npy", "embeddings_fp16.npy")


def read_manifest(path: Path) -> dict:
    """Share duplicate-key and object checks across all artifact consumers."""
    try:
        return parse_record(path.read_bytes(), path)
    except SystemExit as error:
        raise ValueError(f"invalid artifact manifest: {error}") from error


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_export_files(directory: Path, manifest: dict, names=INDEX_PAYLOADS) -> bool:
    """Verify requested payloads, returning False for legacy manifests without hashes."""
    if "schema_version" in manifest and (
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported index manifest schema version")
    hashes = manifest.get("sha256")
    if hashes is None:
        if "schema_version" in manifest:
            raise ValueError("versioned index manifest is missing payload checksums")
        return False
    if not isinstance(hashes, dict) or set(hashes) != set(INDEX_PAYLOADS):
        raise ValueError("index checksums must describe all exported payloads")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("index checksums must be SHA-256 strings")
    for name in names:
        if name not in INDEX_PAYLOADS:
            raise ValueError("unsupported index payload name")
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"index payload must be a regular unlinked file: {name}")
        if file_digest(path) != hashes[name]:
            raise ValueError(f"index payload checksum mismatch: {name}")
    return True


def corpus_ids(corpus: pd.DataFrame) -> np.ndarray:
    if not {"answer_id", "text"} <= set(corpus.columns) or not corpus.columns.is_unique:
        raise ValueError("corpus needs unique answer_id and text columns")
    if corpus.empty:
        raise ValueError("cannot export an empty corpus")
    ids = corpus["answer_id"]
    if (not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any()
            or (ids <= 0).any() or (ids > np.iinfo(np.int64).max).any()):
        raise ValueError("corpus answer IDs must be positive int64 integers")
    if ids.duplicated().any():
        raise ValueError("corpus answer IDs must be unique")
    if not all(isinstance(text, str) and text.strip() for text in corpus["text"]):
        raise ValueError("corpus text must contain nonempty strings")
    return ids.to_numpy(dtype=np.int64)


def validate_document_ids(ids: np.ndarray) -> None:
    if (not isinstance(ids, np.ndarray) or ids.ndim != 1 or not len(ids)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids <= 0) or np.any(ids > np.iinfo(np.int64).max)):
        raise ValueError("document IDs must be a nonempty vector of positive int64 integers")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("document IDs must be unique")


def publish_index(destination: Path, ids: np.ndarray, vectors: np.ndarray, metadata: dict) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("export destination already exists; choose a new version directory")
    validate_document_ids(ids)
    validate_embeddings(vectors, len(ids), atol=1e-4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as staging_name:
        staging = Path(staging_name)
        np.save(staging / "answer_ids.npy", ids.astype(np.int64, copy=False))
        sizes = {}
        for name, dtype in (("fp32", np.float32), ("fp16", np.float16)):
            converted = vectors.astype(dtype, copy=False)
            validate_embeddings(converted, len(ids))
            path = staging / f"embeddings_{name}.npy"
            np.save(path, converted)
            sizes[name] = path.stat().st_size
        manifest = {
            **metadata, "bytes": sizes, "schema_version": 1,
            "documents": len(ids), "dimensions": vectors.shape[1],
            "sha256": {name: file_digest(staging / name) for name in INDEX_PAYLOADS},
        }
        write_result(manifest, staging / "manifest.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("export destination appeared during encoding")
        staging.rename(destination)
    return manifest
