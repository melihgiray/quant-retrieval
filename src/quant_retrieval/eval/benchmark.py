"""Provenance shared by performance reports, independent of relevance metrics."""

import hashlib
import platform
from datetime import UTC, datetime

import pandas as pd

from quant_retrieval.eval.results import current_commit


def benchmark_context(queries: pd.DataFrame, seed: int) -> dict:
    payload = queries[["query_id", "text", "split"]].to_json(orient="records", force_ascii=True)
    return {
        "seed": seed,
        "commit": current_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "query_ids": queries["query_id"].astype(int).tolist(),
        "query_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
