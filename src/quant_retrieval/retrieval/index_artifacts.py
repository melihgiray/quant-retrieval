"""Input and storage contracts for reusable embedding exports."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.eval.results import write_result
from quant_retrieval.retrieval.vectors import validate_embeddings


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


def publish_index(destination: Path, ids: np.ndarray, vectors: np.ndarray, metadata: dict) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("export destination already exists; choose a new version directory")
    validate_embeddings(vectors, len(ids), atol=1e-4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as staging_name:
        staging = Path(staging_name)
        np.save(staging / "answer_ids.npy", ids)
        sizes = {}
        for name, dtype in (("fp32", np.float32), ("fp16", np.float16)):
            converted = vectors.astype(dtype, copy=False)
            validate_embeddings(converted, len(ids))
            path = staging / f"embeddings_{name}.npy"
            np.save(path, converted)
            sizes[name] = path.stat().st_size
        manifest = {**metadata, "bytes": sizes}
        write_result(manifest, staging / "manifest.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("export destination appeared during encoding")
        staging.rename(destination)
    return manifest
