"""Frozen transformer retrieval without sentence-transformers wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from quant_retrieval.models.pooling import POOLING_STRATEGIES, mean_pool, pool
from quant_retrieval.retrieval.base import SearchResult
from quant_retrieval.runtime import choose_device

__all__ = ["DenseRetriever", "choose_device", "mean_pool"]


class DenseRetriever:
    """Cosine retrieval with a frozen Hugging Face encoder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        max_length: int = 256,
        device: str = "auto",
        show_progress: bool = True,
        pooling: str = "mean",
    ) -> None:
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLING_STRATEGIES}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = choose_device(device)
        self.pooling = pooling
        self.show_progress = show_progress
        self.document_ids = np.array([], dtype=np.int64)
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._tokenizer = None
        self._model = None

    def index(self, document_ids: list[int], texts: list[str]) -> None:
        if len(document_ids) != len(texts):
            raise ValueError("document_ids and texts must have the same length")
        if not document_ids:
            raise ValueError("cannot index an empty corpus")
        if any(
            isinstance(document_id, bool)
            or not isinstance(document_id, Integral)
            or document_id <= 0
            for document_id in document_ids
        ):
            raise ValueError("document IDs must be positive integers")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("document texts must be nonempty strings")
        self.document_ids = np.asarray(document_ids, dtype=np.int64)
        self.embeddings = self._encode(texts)

    def load_index(self, document_ids_path: Path, embeddings_path: Path) -> None:
        """Load a precomputed corpus index without copying it into memory."""
        document_ids = np.load(document_ids_path, mmap_mode="r")
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if document_ids.ndim != 1:
            raise ValueError("document IDs must be one-dimensional")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be two-dimensional")
        if len(document_ids) != len(embeddings):
            raise ValueError("document IDs and embeddings must have the same length")
        if not len(document_ids):
            raise ValueError("cannot load an empty index")
        if embeddings.shape[1] == 0:
            raise ValueError("embeddings must have at least one dimension")
        if not np.issubdtype(document_ids.dtype, np.integer):
            raise ValueError("document IDs must be integers")
        if np.any(document_ids <= 0):
            raise ValueError("document IDs must be positive")
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise ValueError("embeddings must be floating point")
        if len(np.unique(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-2, rtol=0):
            raise ValueError("embeddings must be unit normalized")
        self.document_ids = document_ids
        self.embeddings = embeddings

    def validate_query_encoder(self) -> None:
        """Load the model and check its output against the stored index."""
        if not len(self.document_ids):
            raise RuntimeError("load the index before validating the query encoder")
        query_vector = self._encode(["index dimension check"])[0]
        self._validate_query_vector(query_vector)

    def _validate_query_vector(self, query_vector: np.ndarray) -> None:
        if query_vector.shape != (self.embeddings.shape[1],):
            raise ValueError("query encoder dimensions do not match the embeddings")
        if not np.isfinite(query_vector).all():
            raise ValueError("query encoder output must be finite")
        if not np.isclose(np.linalg.norm(query_vector), 1.0, atol=1e-4, rtol=0):
            raise ValueError("query encoder output must be unit normalized")

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if not len(self.document_ids):
            raise RuntimeError("index must be called before search")

        query_embedding = self._encode([query])[0]
        self._validate_query_vector(query_embedding)
        scores = self.embeddings @ query_embedding
        limit = min(k, len(scores))
        candidates = np.argpartition(scores, -limit)[-limit:]
        order = np.lexsort((self.document_ids[candidates], -scores[candidates]))
        rows = candidates[order]
        return [
            SearchResult(document_id=int(self.document_ids[row]), score=float(scores[row]))
            for row in rows
        ]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load_model()
        batches: list[np.ndarray] = []
        starts = range(0, len(texts), self.batch_size)
        show_progress = self.show_progress and len(texts) > self.batch_size
        for start in tqdm(starts, disable=not show_progress, desc="encoding"):
            batch = list(texts[start : start + self.batch_size])
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            with torch.inference_mode():
                output = self._model(**tokens)
                pooled = pool(self.pooling, output.last_hidden_state, tokens["attention_mask"])
                normalized = functional.normalize(pooled, p=2, dim=1)
            batches.append(normalized.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
