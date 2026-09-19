"""Run a retriever against a complete corpus and score its rankings."""

from __future__ import annotations

import time
from collections import defaultdict
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from quant_retrieval.eval.metrics import aggregate_metrics, per_query_metrics
from quant_retrieval.retrieval.base import Retriever


def _require_columns(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


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
    _require_columns(corpus, "corpus", {"answer_id", "text"})
    _require_columns(queries, "queries", {"question_id", "text", "split"})
    _require_columns(qrels, "qrels", {"question_id", "answer_id", "grade"})
    selected_queries = queries.loc[queries["split"] == split]
    if selected_queries.empty:
        raise ValueError(f"split {split!r} contains no queries")
    if (
        not pd.api.types.is_integer_dtype(selected_queries["question_id"].dtype)
        or selected_queries["question_id"].isna().any()
        or (selected_queries["question_id"] <= 0).any()
    ):
        raise ValueError("query IDs must be positive integers")
    if not all(
        isinstance(text, str) and text.strip() for text in selected_queries["text"]
    ):
        raise ValueError("query texts must be nonempty strings")
    if selected_queries["question_id"].duplicated().any():
        raise ValueError(f"split {split!r} contains duplicate question IDs")
    if (
        not pd.api.types.is_integer_dtype(corpus["answer_id"].dtype)
        or corpus["answer_id"].isna().any()
        or (corpus["answer_id"] <= 0).any()
    ):
        raise ValueError("corpus answer IDs must be positive integers")
    if corpus["answer_id"].duplicated().any():
        raise ValueError("corpus contains duplicate answer IDs")
    if not all(isinstance(text, str) and text.strip() for text in corpus["text"]):
        raise ValueError("corpus texts must be nonempty strings")
    selected_ids = set(selected_queries["question_id"])
    selected_qrels = qrels[qrels["question_id"].isin(selected_ids)]
    if (
        not pd.api.types.is_integer_dtype(qrels["question_id"].dtype)
        or qrels["question_id"].isna().any()
        or (qrels["question_id"] <= 0).any()
    ):
        raise ValueError("qrel question IDs must be positive integers")
    if (
        not pd.api.types.is_integer_dtype(qrels["answer_id"].dtype)
        or qrels["answer_id"].isna().any()
        or (qrels["answer_id"] <= 0).any()
    ):
        raise ValueError("qrel answer IDs must be positive integers")
    if (
        not pd.api.types.is_integer_dtype(qrels["grade"].dtype)
        or qrels["grade"].isna().any()
        or not qrels["grade"].isin([1, 2]).all()
    ):
        raise ValueError("qrel grades must be integers in {1, 2}")
    if selected_qrels.duplicated(["question_id", "answer_id"]).any():
        raise ValueError("qrels contain duplicate question and answer pairs")

    document_ids = corpus["answer_id"].astype(int).tolist()
    document_id_set = set(document_ids)
    unknown_qrel_ids = sorted(set(selected_qrels["answer_id"]) - document_id_set)
    if unknown_qrel_ids:
        raise ValueError(f"qrels reference unknown answer IDs: {unknown_qrel_ids[:5]}")
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
        if len(results) > max_results:
            raise ValueError("retriever returned more results than requested")
        if any(
            not isinstance(result.score, Real) or not np.isfinite(result.score)
            for result in results
        ):
            raise ValueError("retriever returned non-finite or non-numeric scores")
        ranking = [result.document_id for result in results]
        if len(set(ranking)) != len(ranking):
            raise ValueError("retriever returned duplicate document IDs")
        unknown = [
            document_id for document_id in ranking if document_id not in document_id_set
        ]
        if unknown:
            raise ValueError(f"retriever returned unknown document IDs: {unknown[:5]}")
        rankings[int(row.question_id)] = ranking

    qrel_map = _qrels_for_queries(qrels, set(rankings))
    if set(qrel_map) != set(rankings):
        missing = sorted(set(rankings) - set(qrel_map))[:5]
        raise ValueError(f"queries have no relevance judgements: {missing}")

    return {
        "metrics": aggregate_metrics(rankings, qrel_map),
        "per_query": per_query_metrics(rankings, qrel_map),
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
