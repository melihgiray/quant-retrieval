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
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant_retrieval.eval.benchmark import benchmark_context  # noqa: E402
from quant_retrieval.eval.results import write_result  # noqa: E402
from quant_retrieval.eval.sampling import sample_queries  # noqa: E402
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


def run_queries(retriever, selected, k: int, watch: Stopwatch, warmup: int, repeats: int):
    if selected.empty or warmup < 0 or repeats <= 0 or k <= 0:
        raise ValueError("queries and repeats must be positive; warmup must be nonnegative")
    texts = selected["text"].tolist()
    for index in range(warmup):
        retriever.search(texts[index % len(texts)], k)
    watch.samples.clear()
    timings = []
    for repetition in range(repeats):
        for row in selected.itertuples(index=False):
            started = time.perf_counter()
            retriever.search(row.text, k)
            timings.append({"question_id": int(row.question_id), "repeat": repetition + 1,
                            "latency_ms": (time.perf_counter() - started) * 1000})
    return timings


@contextmanager
def instrument(retriever: Any, watch: Stopwatch, label: str):
    """Time each distinct node and restore every method when profiling ends."""
    originals = []
    seen = set()

    def wrap(node, method, name):
        original = getattr(node, method)
        originals.append((node, method, method in vars(node), original))

        def timed(*args, **kwargs):
            started = time.perf_counter()
            results = original(*args, **kwargs)
            watch.record(name, time.perf_counter() - started)
            return results

        setattr(node, method, timed)

    def visit(node, name):
        if id(node) in seen:
            return
        seen.add(id(node))
        wrap(node, "search", name)
        if isinstance(node, HybridRetriever):
            for index, child in enumerate(node.retrievers):
                visit(child, f"{name}.child{index}:{type(child).__name__}")
        elif isinstance(node, RerankingRetriever):
            visit(node.base, f"{name}.base:{type(node.base).__name__}")
            wrap(node, "_score", f"{name}.cross_encoder")

    try:
        visit(retriever, label)
        yield retriever
    finally:
        for node, method, was_local, original in reversed(originals):
            if was_local:
                setattr(node, method, original)
            else:
                delattr(node, method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.queries <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("queries/repeats must be positive and warmup nonnegative")

    config = yaml.safe_load(args.config.read_text())
    seed = int(config.get("seed", 17))
    set_seed(seed)

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    selected = sample_queries(queries, args.queries, seed, config.get("split", "val"))

    watch = Stopwatch()
    retriever = build_retriever(config)

    started = time.perf_counter()
    retriever.index(corpus["answer_id"].astype(int).tolist(), corpus["text"].tolist())
    index_seconds = time.perf_counter() - started

    max_results = int(config.get("max_results", 100))
    with instrument(retriever, watch, config["retriever"]):
        query_timings = run_queries(
            retriever, selected, max_results, watch, args.warmup, args.repeats
        )

    report = {
        **benchmark_context(selected, seed),
        "config": str(args.config),
        "configuration": config,
        "split": config.get("split", "val"),
        "queries": len(selected),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measured_calls": len(selected) * args.repeats,
        "max_results": max_results,
        "corpus_documents": len(corpus),
        "index_seconds": round(index_seconds, 2),
        "stages": watch.report(),
        "query_timings": query_timings,
    }

    output = args.output or Path("results") / f"{args.config.stem}_profile.json"
    write_result(report, output)

    print(f"wrote {output}")
    print(f"index: {index_seconds:.2f}s for {len(corpus)} documents")
    for label, stats in report["stages"].items():
        print(f"{label:<44} p50 {stats['p50_ms']:>8.2f}ms  p95 {stats['p95_ms']:>8.2f}ms")
    print()
    print("Child rows are included in their parent, so they do not sum to it.")


if __name__ == "__main__":
    main()
