"""Load the measured retrieval pipeline and turn rankings into display records."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import pandas as pd

from quant_retrieval.retrieval.base import Retriever
from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.hybrid import HybridRetriever


def make_snippet(text: str, max_chars: int = 600) -> str:
    """Collapse whitespace and stop at a word boundary for browser responses."""
    if max_chars < 2:
        raise ValueError("max_chars must be at least 2")
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    prefix = compact[: max_chars - 1].rsplit(" ", 1)[0]
    return f"{prefix or compact[: max_chars - 1]}…"


@dataclass(frozen=True)
class SearchHit:
    answer_id: int
    question_id: int
    score: float
    text: str
    url: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactManifest:
    documents: int
    dimensions: int
    max_length: int

    @classmethod
    def load(cls, path: Path) -> ArtifactManifest:
        try:
            payload = json.loads(path.read_text())
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("artifact manifest is not valid JSON") from error
        try:
            if not isinstance(payload, dict) or any(
                type(payload.get(name)) is not int
                for name in ("documents", "dimensions", "max_length")
            ):
                raise ValueError("artifact manifest is missing valid dimensions")
            manifest = cls(
                documents=payload["documents"],
                dimensions=payload["dimensions"],
                max_length=payload["max_length"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("artifact manifest is missing valid dimensions") from error
        if min(manifest.documents, manifest.dimensions, manifest.max_length) < 1:
            raise ValueError("artifact manifest values must be positive")
        return manifest


class SearchService:
    """Attach answer text and source links to a retriever's ranked IDs."""

    def __init__(self, retriever: Retriever, corpus: pd.DataFrame) -> None:
        self._validate_corpus(corpus)
        self.retriever = retriever
        self._search_lock = Lock()
        self.answers = {
            int(row.answer_id): (int(row.question_id), str(row.text))
            for row in corpus.itertuples(index=False)
        }

    @staticmethod
    def _validate_corpus(corpus: pd.DataFrame) -> None:
        required = {"answer_id", "question_id", "text"}
        missing = required - set(corpus.columns)
        if missing:
            raise ValueError(f"corpus is missing columns: {sorted(missing)}")
        for name in ("answer_id", "question_id"):
            values = corpus[name]
            if (
                not pd.api.types.is_integer_dtype(values.dtype)
                or values.isna().any()
                or (values <= 0).any()
            ):
                raise ValueError(f"corpus {name} values must be positive integers")
        if not all(isinstance(text, str) and text.strip() for text in corpus["text"]):
            raise ValueError("corpus text values must be nonempty strings")
        if corpus["answer_id"].duplicated().any():
            raise ValueError("corpus answer IDs must be unique")

    @property
    def document_count(self) -> int:
        return len(self.answers)

    @property
    def pipeline(self) -> str:
        return "bm25_dense_rrf"

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint: Path,
        corpus_path: Path,
        manifest_path: Path,
        document_ids_path: Path,
        embeddings_path: Path,
        device: str = "auto",
        depth: int = 100,
        rrf_k: int = 60,
    ) -> SearchService:
        manifest = ArtifactManifest.load(manifest_path)
        corpus = pd.read_parquet(corpus_path)
        cls._validate_corpus(corpus)
        ids = corpus["answer_id"].astype(int).tolist()
        texts = corpus["text"].astype(str).tolist()

        bm25 = BM25Retriever()
        bm25.index(ids, texts)
        dense = DenseRetriever(
            str(checkpoint),
            device=device,
            max_length=manifest.max_length,
            show_progress=False,
        )
        dense.load_index(document_ids_path, embeddings_path)
        if manifest.documents != len(corpus) or manifest.documents != len(dense.document_ids):
            raise ValueError("artifact manifest document count does not match the index")
        if manifest.dimensions != dense.embeddings.shape[1]:
            raise ValueError("artifact manifest dimensions do not match the embeddings")
        if set(dense.document_ids.tolist()) != set(ids):
            raise ValueError("precomputed index does not match the corpus")
        dense.validate_query_encoder()

        retriever = HybridRetriever([bm25, dense], depth=depth, rrf_k=rrf_k)
        return cls(retriever, corpus)

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")

        with self._search_lock:
            ranked = self.retriever.search(query, k)
        if len(ranked) > k:
            raise RuntimeError(f"retriever returned {len(ranked)} answers for a limit of {k}")

        hits = []
        seen: set[int] = set()
        previous_score = math.inf
        for result in ranked:
            if result.document_id in seen:
                raise RuntimeError(f"retriever returned duplicate answer {result.document_id}")
            seen.add(result.document_id)
            if not math.isfinite(result.score):
                raise RuntimeError(
                    f"retriever returned a non-finite score for {result.document_id}"
                )
            if result.score > previous_score:
                raise RuntimeError("retriever results are not in descending score order")
            previous_score = result.score
            if result.document_id not in self.answers:
                raise RuntimeError(f"retriever returned unknown answer {result.document_id}")
            question_id, text = self.answers[result.document_id]
            hits.append(
                SearchHit(
                    answer_id=result.document_id,
                    question_id=question_id,
                    score=result.score,
                    text=make_snippet(text),
                    url=f"https://quant.stackexchange.com/a/{result.document_id}",
                )
            )
        return hits
