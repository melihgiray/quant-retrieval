"""Deployment settings with local paths as explicit defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _positive_setting(values: Mapping[str, str], name: str, default: int) -> int:
    try:
        parsed = int(values.get(name, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class ServeSettings:
    model_path: Path = Path("checkpoints/minilm_tuned/epoch-3")
    corpus_path: Path = Path("data/processed/corpus.parquet")
    manifest_path: Path = Path("artifacts/manifest.json")
    document_ids_path: Path = Path("artifacts/answer_ids.npy")
    embeddings_path: Path = Path("artifacts/embeddings_fp16.npy")
    device: str = "auto"
    depth: int = 100
    rrf_k: int = 60

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ServeSettings:
        values = os.environ if environ is None else environ
        settings = cls(
            model_path=Path(values.get("MODEL_PATH", cls.model_path)),
            corpus_path=Path(values.get("CORPUS_PATH", cls.corpus_path)),
            manifest_path=Path(values.get("MANIFEST_PATH", cls.manifest_path)),
            document_ids_path=Path(values.get("DOCUMENT_IDS_PATH", cls.document_ids_path)),
            embeddings_path=Path(values.get("EMBEDDINGS_PATH", cls.embeddings_path)),
            device=values.get("DEVICE", cls.device),
            depth=_positive_setting(values, "RETRIEVAL_DEPTH", cls.depth),
            rrf_k=_positive_setting(values, "RRF_K", cls.rrf_k),
        )
        return settings
