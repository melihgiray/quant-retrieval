"""Shared interface and result type for retrieval systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    score: float


class Retriever(Protocol):
    def index(self, document_ids: list[int], texts: list[str]) -> None:
        """Build whatever state the retriever needs over the full corpus."""

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return at most ``k`` documents in descending score order."""
