"""Find the corpus size where approximate search starts to pay.

    python scripts/ann_sweep.py --embeddings artifacts artifacts/scale_100000

For each set of embeddings and each value of ef_search, this measures two
things: how much of exact search's top 10 the graph still finds, and how long a
query takes. Exact search over the same vectors is the reference for both.

The comparison is deliberately narrow. Both sides read the same embeddings from
the same file, so the model is held fixed and the only variable is the index.
That keeps a lossy index from being confused with a bad model, which is the way
this measurement usually goes wrong.

Queries are encoded once and reused across every setting. Re-encoding them per
sweep point would add the encoder's time to the index's and make the fast
settings look slower than they are.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quant_retrieval.retrieval.ann import ApproximateRetriever, recall_against_exact  # noqa: E402
from quant_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from quant_retrieval.runtime import set_seed  # noqa: E402

DEFAULT_EF_SEARCH = (16, 32, 64, 128, 256)


def time_search(retriever, queries: np.ndarray, k: int) -> tuple[list, list[float]]:
    """Run every query once, keeping the results and the per query latency."""
    results, latencies = [], []
    for vector in queries:
        started = time.perf_counter()
        results.append(retriever.search_vector(vector, k))
        latencies.append((time.perf_counter() - started) * 1000)
    return results, latencies


def summarise(latencies: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="+", type=Path, required=True,
                        help="directories written by scripts/export_index.py")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/minilm_tuned/epoch-3"))
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", nargs="+", type=int, default=list(DEFAULT_EF_SEARCH))
    parser.add_argument("--neighbours", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("results/ann_scaling.json"))
    args = parser.parse_args()

    set_seed(args.seed)

    # Encode the queries once, on whatever device is available, then never again.
    queries = pd.read_parquet(args.data / "queries.parquet")
    selected = queries[queries["split"] == "val"].head(args.queries)
    encoder = DenseRetriever(str(args.checkpoint), show_progress=False)
    query_vectors = encoder._encode(selected["text"].tolist())
    print(f"encoded {len(query_vectors)} queries on {encoder.device}")

    runs = []
    for directory in args.embeddings:
        manifest = json.loads((directory / "manifest.json").read_text())
        answer_ids = np.load(directory / "answer_ids.npy").tolist()
        path = directory / "embeddings_fp32.npy"
        documents = manifest["documents"]
        print(f"\n=== {documents} documents from {directory} ===")

        exact = ApproximateRetriever(path, exact=True)
        exact.index(answer_ids, [])
        exact_results, exact_latencies = time_search(exact, query_vectors, args.k)
        runs.append(
            {
                "documents": documents,
                "index": "exact",
                "ef_search": None,
                "recall_at_k": 1.0,
                **summarise(exact_latencies),
            }
        )
        print(f"exact      p50 {runs[-1]['p50_ms']:>7.3f}ms  recall 1.000")

        for ef_search in args.ef_search:
            approximate = ApproximateRetriever(
                path, neighbours=args.neighbours, ef_search=ef_search
            )
            build_started = time.perf_counter()
            approximate.index(answer_ids, [])
            build_seconds = time.perf_counter() - build_started

            results, latencies = time_search(approximate, query_vectors, args.k)
            recall = float(
                np.mean(
                    [
                        recall_against_exact(got, want, args.k)
                        for got, want in zip(results, exact_results, strict=True)
                    ]
                )
            )
            runs.append(
                {
                    "documents": documents,
                    "index": "hnsw",
                    "ef_search": ef_search,
                    "neighbours": args.neighbours,
                    "recall_at_k": round(recall, 4),
                    "build_seconds": round(build_seconds, 1),
                    **summarise(latencies),
                }
            )
            print(
                f"hnsw ef={ef_search:<4} p50 {runs[-1]['p50_ms']:>7.3f}ms  "
                f"recall {recall:.3f}  build {build_seconds:.0f}s"
            )

    report = {"k": args.k, "queries": len(query_vectors), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")

    # The headline: at each size, the fastest setting that keeps recall high.
    print("\nfastest HNSW setting reaching 0.95 recall, against exact:")
    for documents in sorted({run["documents"] for run in runs}):
        at_size = [r for r in runs if r["documents"] == documents]
        exact_p50 = next(r["p50_ms"] for r in at_size if r["index"] == "exact")
        good = [r for r in at_size if r["index"] == "hnsw" and r["recall_at_k"] >= 0.95]
        if not good:
            print(f"{documents:>8} documents: nothing reached 0.95, exact {exact_p50:.2f}ms")
            continue
        best = min(good, key=lambda r: r["p50_ms"])
        verdict = "HNSW wins" if best["p50_ms"] < exact_p50 else "exact still wins"
        print(
            f"{documents:>8} documents: ef={best['ef_search']:<4} "
            f"{best['p50_ms']:.2f}ms against exact {exact_p50:.2f}ms   {verdict}"
        )


if __name__ == "__main__":
    main()
