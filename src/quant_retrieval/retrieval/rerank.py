"""Reorder a shortlist with the cross-encoder.

Retrieve wide and cheap, then rescore narrow and expensive. The base retriever
sees all 26,152 documents and is allowed to be approximate about it; the
cross-encoder sees `depth` of them and reads each one properly.

Query and document are trimmed with `longest_first`, matching training. A
mismatch there would score documents with a different input shape than the one
the model learned on.

`depth` is the whole trade. Everything past it is unreachable no matter how good
the reranker is, so the ceiling on this pipeline is the base retriever's
Recall@depth, not its ranking. That is why Recall@100 has been the number to
watch since the baselines went in.
"""

from __future__ import annotations

import numpy as np
import torch

from quant_retrieval.models.cross_encoder import CrossEncoder
from quant_retrieval.retrieval.base import Retriever, SearchResult
from quant_retrieval.runtime import choose_device


class RerankingRetriever:
    """A base retriever with a cross-encoder second stage."""

    def __init__(
        self,
        base: Retriever,
        model_name: str,
        depth: int = 50,
        batch_size: int = 64,
        max_length: int = 320,
        device: str = "auto",
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.base = base
        self.model_name = model_name
        self.depth = depth
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = choose_device(device)
        self.documents: dict[int, str] = {}
        self._model = None
        self._tokenizer = None

    def index(self, document_ids: list[int], texts: list[str]) -> None:
        self.base.index(document_ids, texts)
        # Kept because reranking needs the text back, not just the id.
        self.documents = dict(zip(document_ids, texts, strict=True))

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")

        shortlist = self.base.search(query, max(k, self.depth))
        head, tail = shortlist[: self.depth], shortlist[self.depth :]
        if not head:
            return []

        scores = self._score(query, [self.documents[r.document_id] for r in head])
        order = np.lexsort((np.array([r.document_id for r in head]), -scores))
        reranked = [
            SearchResult(document_id=head[i].document_id, score=float(scores[i])) for i in order
        ]

        # Anything past `depth` keeps its original order and sits below every
        # reranked document. It was never rescored, so it cannot be interleaved
        # honestly, and dropping it would break Recall@100 for k above depth.
        return (reranked + list(tail))[:k]

    def _score(self, query: str, documents: list[str]) -> np.ndarray:
        self._load_model()
        scores: list[np.ndarray] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            tokens = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation="longest_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            with torch.inference_mode():
                scores.append(self._model(**tokens).cpu().numpy().astype(np.float32))
        return np.concatenate(scores)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from pathlib import Path

        from transformers import AutoTokenizer

        self._model = CrossEncoder.load(Path(self.model_name))
        self._model.eval()
        self._model.to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
