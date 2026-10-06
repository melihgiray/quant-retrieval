"""Provenance shared by performance reports, independent of relevance metrics."""

import hashlib
import platform
from datetime import UTC, datetime

import pandas as pd

from quant_retrieval.eval.results import current_commit


def runtime_context() -> dict:
    """Describe the process taking measurements, independently of where queries were encoded."""
    return {
        "commit": current_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def benchmark_context(queries: pd.DataFrame, seed: int) -> dict:
    payload = queries[["question_id", "text", "split"]].to_json(orient="records", force_ascii=True)
    return {**runtime_context(), "seed": seed,
            "question_ids": queries["question_id"].astype(int).tolist(),
            "query_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
