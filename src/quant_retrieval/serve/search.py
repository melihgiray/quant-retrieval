"""Load the measured retrieval pipeline and turn rankings into display records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from quant_retrieval.retrieval.base import Retriever
from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class SearchHit:
    answer_id: int
    question_id: int
    score: float
    text: str
    url: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class SearchService:
    """Attach answer text and source links to a retriever's ranked IDs."""

    def __init__(self, retriever: Retriever, corpus: pd.DataFrame) -> None:
        required = {"answer_id", "question_id", "text"}
        missing = required - set(corpus.columns)
        if missing:
            raise ValueError(f"corpus is missing columns: {sorted(missing)}")
        if corpus["answer_id"].duplicated().any():
            raise ValueError("corpus answer IDs must be unique")
        self.retriever = retriever
        self.answers = {
            int(row.answer_id): (int(row.question_id), str(row.text))
            for row in corpus.itertuples(index=False)
        }

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint: Path,
        corpus_path: Path,
        document_ids_path: Path,
        embeddings_path: Path,
        device: str = "auto",
        depth: int = 100,
        rrf_k: int = 60,
    ) -> SearchService:
        corpus = pd.read_parquet(corpus_path)
        ids = corpus["answer_id"].astype(int).tolist()
        texts = corpus["text"].astype(str).tolist()

        bm25 = BM25Retriever()
        bm25.index(ids, texts)
        dense = DenseRetriever(str(checkpoint), device=device, show_progress=False)
        dense.load_index(document_ids_path, embeddings_path)
        if set(dense.document_ids.tolist()) != set(ids):
            raise ValueError("precomputed index does not match the corpus")

        retriever = HybridRetriever([bm25, dense], depth=depth, rrf_k=rrf_k)
        return cls(retriever, corpus)

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")

        hits = []
        for result in self.retriever.search(query, k):
            if result.document_id not in self.answers:
                raise RuntimeError(f"retriever returned unknown answer {result.document_id}")
            question_id, text = self.answers[result.document_id]
            hits.append(
                SearchHit(
                    answer_id=result.document_id,
                    question_id=question_id,
                    score=result.score,
                    text=text,
                    url=f"https://quant.stackexchange.com/a/{result.document_id}",
                )
            )
        return hits
