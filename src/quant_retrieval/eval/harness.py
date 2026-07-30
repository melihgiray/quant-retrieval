"""Run a retriever against a complete corpus and score its rankings."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from quant_retrieval.eval.metrics import aggregate_metrics
from quant_retrieval.retrieval.base import Retriever


def evaluate_retriever(
    retriever: Retriever,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    split: str = "val",
    max_results: int = 100,
) -> dict[str, Any]:
    """Index the full corpus, retrieve one ranking per query, and score it."""
    if max_results < 100:
        raise ValueError("max_results must be at least 100 for Recall@100")
    selected_queries = queries.loc[queries["split"] == split]
    if selected_queries.empty:
        raise ValueError(f"split {split!r} contains no queries")

    document_ids = corpus["answer_id"].astype(int).tolist()
    document_texts = corpus["text"].tolist()
    index_started = time.perf_counter()
    retriever.index(document_ids, document_texts)
    index_seconds = time.perf_counter() - index_started

    rankings: dict[int, list[int]] = {}
    latencies_ms: list[float] = []
    for row in selected_queries.itertuples(index=False):
        search_started = time.perf_counter()
        results = retriever.search(row.text, max_results)
        latencies_ms.append((time.perf_counter() - search_started) * 1000)
        rankings[int(row.question_id)] = [result.document_id for result in results]

    qrel_map = _qrels_for_queries(qrels, set(rankings))
    if set(qrel_map) != set(rankings):
        missing = sorted(set(rankings) - set(qrel_map))[:5]
        raise ValueError(f"queries have no relevance judgements: {missing}")

    return {
        "metrics": aggregate_metrics(rankings, qrel_map),
        "timing": {
            "index_seconds": index_seconds,
            "search_total_seconds": sum(latencies_ms) / 1000,
            "search_ms_per_query_p50": float(np.percentile(latencies_ms, 50)),
            "search_ms_per_query_p95": float(np.percentile(latencies_ms, 95)),
        },
        "counts": {
            "corpus_documents": len(corpus),
            "queries": len(selected_queries),
            "max_results": max_results,
        },
        "rankings": rankings,
    }


def _qrels_for_queries(
    qrels: pd.DataFrame, query_ids: set[int]
) -> dict[int, dict[int, int]]:
    relevant: dict[int, dict[int, int]] = defaultdict(dict)
    rows = qrels[qrels["question_id"].isin(query_ids)]
    for row in rows.itertuples(index=False):
        relevant[int(row.question_id)][int(row.answer_id)] = int(row.grade)
    return dict(relevant)
