"""Portable validation-query vectors for model-free index benchmarks."""

import re
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from quant_retrieval.eval.results import write_result
from quant_retrieval.retrieval.checkpoint import validate_checkpoint_hashes
from quant_retrieval.retrieval.index_artifacts import file_digest, read_manifest
from quant_retrieval.retrieval.vectors import validate_embeddings

QUERY_VECTOR_FILE = "query_vectors.npy"


def validate_query_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("kind") != "ann_queries"):
        raise ValueError("unsupported query artifact schema")
    if manifest.get("split") != "val":
        raise ValueError("ANN query artifacts must use the validation split")
    for name in ("queries", "dimensions", "max_length"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"query artifact {name} must be a positive integer")
    if manifest.get("pooling") not in ("mean", "cls") or manifest.get("dtype") != "float32":
        raise ValueError("query artifacts require supported pooling and float32 vectors")
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("query artifact seed must fit uint32")
    ids = manifest.get("question_ids")
    if (not isinstance(ids, list) or len(ids) != manifest["queries"]
            or any(type(value) is not int or not 0 < value < 2**63 for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("query artifact IDs must be unique positive int64 values matching rows")
    for name in ("query_sha256", "vectors_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"query artifact {name} must be a SHA-256 string")
    validate_checkpoint_hashes(manifest.get("checkpoint_sha256"))


def publish_queries(destination: Path, vectors: np.ndarray, metadata: dict) -> dict:
    """Publish a complete new query version without replacing an earlier one."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("query destination exists; choose a new version directory")
    validate_embeddings(vectors, len(metadata.get("question_ids", [])), atol=1e-4)
    if vectors.dtype != np.float32:
        raise ValueError("saved query vectors must be float32")
    manifest = {**deepcopy(metadata), "schema_version": 1, "kind": "ann_queries",
                "queries": len(vectors), "dimensions": vectors.shape[1], "dtype": "float32",
                "vectors_sha256": "0" * 64}
    validate_query_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,
                                     prefix=f".{destination.name}.") as temporary:
        staging = Path(temporary)
        np.save(staging / QUERY_VECTOR_FILE, vectors)
        manifest["vectors_sha256"] = file_digest(staging / QUERY_VECTOR_FILE)
        write_result(manifest, staging / "manifest.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("query destination appeared during publication")
        staging.rename(destination)
    return manifest


def load_queries(directory: Path) -> tuple[dict, np.ndarray]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("query artifact must be a regular directory")
    for name in ("manifest.json", QUERY_VECTOR_FILE):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"query artifact needs an unlinked regular file: {name}")
    manifest = read_manifest(directory / "manifest.json")
    validate_query_manifest(manifest)
    path = directory / QUERY_VECTOR_FILE
    if file_digest(path) != manifest["vectors_sha256"]:
        raise ValueError("query vector checksum mismatch")
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    validate_embeddings(vectors, manifest["queries"], atol=1e-4)
    if vectors.dtype != np.float32 or vectors.shape[1] != manifest["dimensions"]:
        raise ValueError("query vectors disagree with manifest dtype or dimensions")
    return manifest, vectors
