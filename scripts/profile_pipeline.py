"""Time each stage of a retrieval pipeline separately.

    python scripts/profile_pipeline.py --config configs/hybrid.yaml --queries 100

The reason this exists: hybrid search measures 60.85ms per query while its two
halves measure 5.64ms and 4.89ms standalone at the same depth. About 50ms is
unaccounted for, and an aggregate number cannot say where it went. This walks the
tree the config describes and times every level, so the answer is measured rather
than guessed.

Indexing is timed too. It is not part of query latency, but it is what a cold
start pays, which is the number that matters for a small hosted demo.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant_retrieval.retrieval.factory import build_retriever  # noqa: E402
from quant_retrieval.retrieval.hybrid import HybridRetriever  # noqa: E402
from quant_retrieval.retrieval.rerank import RerankingRetriever  # noqa: E402
from quant_retrieval.runtime import set_seed  # noqa: E402


class Stopwatch:
    """Collects per call durations, keyed by a label."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def record(self, label: str, seconds: float) -> None:
        self.samples.setdefault(label, []).append(seconds * 1000)

    def report(self) -> dict[str, dict[str, float]]:
        return {
            label: {
                "calls": len(values),
                "p50_ms": round(float(np.percentile(values, 50)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
                "total_ms": round(float(np.sum(values)), 1),
            }
            for label, values in sorted(self.samples.items())
        }


def instrument(retriever: Any, watch: Stopwatch, label: str) -> Any:
    """Wrap `search` so every level of the tree reports its own time.

    Wrapping rather than editing the retrievers keeps the timing out of the
    production path. A profiler that changes what it measures is worth little.
    """
    original = retriever.search

    def timed(query: str, k: int):
        started = time.perf_counter()
        results = original(query, k)
        watch.record(label, time.perf_counter() - started)
        return results

    retriever.search = timed

    if isinstance(retriever, HybridRetriever):
        for index, child in enumerate(retriever.retrievers):
            instrument(child, watch, f"{label}.child{index}:{type(child).__name__}")
    elif isinstance(retriever, RerankingRetriever):
        instrument(retriever.base, watch, f"{label}.base:{type(retriever.base).__name__}")
        original_score = retriever._score

        def timed_score(query: str, documents: list[str]):
            started = time.perf_counter()
            scores = original_score(query, documents)
            watch.record(f"{label}.cross_encoder", time.perf_counter() - started)
            return scores

        retriever._score = timed_score

    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    set_seed(int(config.get("seed", 17)))

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    selected = queries[queries["split"] == config.get("split", "val")].head(args.queries)

    watch = Stopwatch()
    retriever = instrument(
        build_retriever(config), watch, config["retriever"]
    )

    started = time.perf_counter()
    retriever.index(corpus["answer_id"].astype(int).tolist(), corpus["text"].tolist())
    index_seconds = time.perf_counter() - started

    max_results = int(config.get("max_results", 100))
    for row in selected.itertuples(index=False):
        retriever.search(row.text, max_results)

    report = {
        "config": str(args.config),
        "queries": len(selected),
        "max_results": max_results,
        "corpus_documents": len(corpus),
        "index_seconds": round(index_seconds, 2),
        "stages": watch.report(),
    }

    output = args.output or Path("results") / f"{args.config.stem}_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output}")
    print(f"index: {index_seconds:.2f}s for {len(corpus)} documents")
    for label, stats in report["stages"].items():
        print(f"{label:<44} p50 {stats['p50_ms']:>8.2f}ms  p95 {stats['p95_ms']:>8.2f}ms")
    print()
    print("Child rows are included in their parent, so they do not sum to it.")


if __name__ == "__main__":
    main()
