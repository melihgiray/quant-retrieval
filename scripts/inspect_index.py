"""Inspect index integrity and float16 storage error without loading a model."""

import argparse
from pathlib import Path

import numpy as np

from quant_retrieval.eval.results import parse_record, write_result
from quant_retrieval.retrieval.index_artifacts import file_digest, verify_export_files
from quant_retrieval.retrieval.vectors import validate_embeddings


def inspect_index(directory: Path, chunk_size: int = 4096) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = parse_record(manifest_path.read_bytes(), manifest_path)
    verified = verify_export_files(directory, manifest)
    ids = np.load(directory / "answer_ids.npy", mmap_mode="r")
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or np.any(ids <= 0) or np.any(ids > np.iinfo(np.int64).max)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("index IDs must be unique positive int64 values")
    matrices, storage = {}, {}
    for name, dtype in (("fp32", np.float32), ("fp16", np.float16)):
        path = directory / f"embeddings_{name}.npy"
        vectors = np.load(path, mmap_mode="r")
        if vectors.dtype != dtype:
            raise ValueError(f"{name} payload has the wrong stored dtype")
        validate_embeddings(vectors, len(ids), chunk_size=chunk_size)
        if vectors.shape != (manifest.get("documents"), manifest.get("dimensions")):
            raise ValueError("manifest dimensions do not match the saved vectors")
        matrices[name] = vectors
        storage[name] = {"dtype": str(vectors.dtype), "array_bytes": vectors.nbytes,
                         "file_bytes": path.stat().st_size}
    squared_error = maximum_error = norm_error = 0.0
    for start in range(0, len(ids), chunk_size):
        full = matrices["fp32"][start:start + chunk_size].astype(np.float64)
        half = matrices["fp16"][start:start + chunk_size].astype(np.float64)
        delta = half - full
        squared_error += float(np.sum(delta * delta))
        maximum_error = max(maximum_error, float(np.max(np.abs(delta))))
        norm_error = max(norm_error, float(np.max(np.abs(np.linalg.norm(half, axis=1) - 1))))
    return {
        "documents": len(ids), "dimensions": matrices["fp32"].shape[1],
        "payload_checksums_verified": verified,
        "manifest_sha256": file_digest(manifest_path), "storage": storage,
        "fp16_vs_fp32": {"max_absolute_error": maximum_error,
                         "root_mean_square_error": float(np.sqrt(
                             squared_error / matrices["fp32"].size)),
                         "max_unit_norm_error": norm_error},
        "scope": "storage and numerical error, not retrieval quality",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(inspect_index(args.artifacts), args.output, overwrite=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
