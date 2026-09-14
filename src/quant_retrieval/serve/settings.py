"""Deployment settings with local paths as explicit defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
            depth=int(values.get("RETRIEVAL_DEPTH", cls.depth)),
            rrf_k=int(values.get("RRF_K", cls.rrf_k)),
        )
        if settings.depth < 1:
            raise ValueError("RETRIEVAL_DEPTH must be positive")
        if settings.rrf_k < 1:
            raise ValueError("RRF_K must be positive")
        return settings
