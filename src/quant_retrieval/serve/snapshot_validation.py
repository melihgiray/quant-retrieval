"""Offline consistency checks for deployment assets, without loading the encoder."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.index_artifacts import corpus_ids


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
