"""Frozen transformer retrieval without sentence-transformers wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from quant_retrieval.models.pooling import mean_pool
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
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = choose_device(device)
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
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")
        self.document_ids = np.asarray(document_ids, dtype=np.int64)
        self.embeddings = self._encode(texts)

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not len(self.document_ids):
            raise RuntimeError("index must be called before search")

        query_embedding = self._encode([query])[0]
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
                pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
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
