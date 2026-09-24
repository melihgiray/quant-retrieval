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

from quant_retrieval.eval.benchmark import benchmark_context  # noqa: E402
from quant_retrieval.eval.results import write_result  # noqa: E402
from quant_retrieval.eval.sampling import sample_queries  # noqa: E402
from quant_retrieval.retrieval.ann import ApproximateRetriever, recall_against_exact  # noqa: E402
from quant_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from quant_retrieval.runtime import set_seed  # noqa: E402

DEFAULT_EF_SEARCH = (16, 32, 64, 128, 256)


def load_manifest(directory: Path, checkpoint: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for key in ("documents", "dimensions", "max_length"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError(f"{directory}: {key} must be a positive integer")
    if not isinstance(manifest.get("checkpoint"), str) or not manifest["checkpoint"]:
        raise ValueError(f"{directory}: checkpoint is required")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{directory}: checkpoint does not match the query encoder")
    ids = np.load(directory / "answer_ids.npy", mmap_mode="r")
    vectors = np.load(directory / "embeddings_fp32.npy", mmap_mode="r")
    if ids.shape != (manifest["documents"],):
        raise ValueError(f"{directory}: document count disagrees with answer IDs")
    if vectors.shape != (manifest["documents"], manifest["dimensions"]):
        raise ValueError(f"{directory}: vector shape disagrees with manifest")
    return manifest


def time_search(
    retriever, queries: np.ndarray, k: int, warmup: int = 0
) -> tuple[list, list[float]]:
    """Warm the index, then keep one measured result per query."""
    if warmup < 0 or len(queries) == 0:
        raise ValueError("warmup must be nonnegative and queries must not be empty")
    for index in range(warmup):
        retriever.search_vector(queries[index % len(queries)], k)
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


def best_settings(runs: list[dict], target: float) -> list[dict]:
    """Compare settings only within the same artifact, not just the same size."""
    summaries = []
    for artifact in dict.fromkeys(run["artifact"] for run in runs):
        group = [run for run in runs if run["artifact"] == artifact]
        exact = next(run for run in group if run["index"] == "exact")
        eligible = [r for r in group if r["index"] == "hnsw" and r["recall_at_k"] >= target]
        summaries.append({"artifact": artifact, "documents": exact["documents"],
                          "exact_p50_ms": exact["p50_ms"],
                          "best": min(eligible, key=lambda r: r["p50_ms"], default=None)})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="+", type=Path, required=True,
                        help="directories written by scripts/export_index.py")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/minilm_tuned/epoch-3"))
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ef-search", nargs="+", type=int, default=list(DEFAULT_EF_SEARCH))
    parser.add_argument("--neighbours", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1, help="FAISS CPU threads")
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("results/ann_scaling.json"))
    args = parser.parse_args()
    if any(value <= 0 for value in [args.queries, args.k, args.neighbours,
                                    args.ef_construction, args.threads, *args.ef_search]):
        parser.error("query counts, graph settings and threads must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not 0 <= args.recall_target <= 1:
        parser.error("recall target must lie between zero and one")

    import faiss

    faiss.omp_set_num_threads(args.threads)

    set_seed(args.seed)
    manifests = [load_manifest(directory, args.checkpoint) for directory in args.embeddings]
    if len({(m["dimensions"], m["max_length"]) for m in manifests}) != 1:
        parser.error("all embedding sets must use the same dimensions and max_length")

    # Encode the queries once, on whatever device is available, then never again.
    queries = pd.read_parquet(args.data / "queries.parquet")
    selected = sample_queries(queries, args.queries, args.seed)
    encoder = DenseRetriever(
        str(args.checkpoint), max_length=manifests[0]["max_length"], show_progress=False
    )
    query_vectors = encoder._encode(selected["text"].tolist())
    if query_vectors.shape != (len(selected), manifests[0]["dimensions"]):
        raise ValueError("query encoder dimensions disagree with exported vectors")
    print(f"encoded {len(query_vectors)} queries on {encoder.device}")

    runs = []
    report = {
        **benchmark_context(selected, args.seed),
        "artifacts": [
            {"directory": str(directory.resolve()), "manifest": manifest}
            for directory, manifest in zip(args.embeddings, manifests, strict=True)
        ],
        "checkpoint": str(args.checkpoint),
        "k": args.k, "queries": len(query_vectors), "warmup": args.warmup,
        "threads": args.threads, "neighbours": args.neighbours,
        "ef_construction": args.ef_construction, "ef_search": args.ef_search,
        "complete": False, "runs": runs,
        "recall_target": args.recall_target,
    }
    write_result(report, args.output)
    for directory, manifest in zip(args.embeddings, manifests, strict=True):
        answer_ids = np.load(directory / "answer_ids.npy").tolist()
        path = directory / "embeddings_fp32.npy"
        documents = manifest["documents"]
        print(f"\n=== {documents} documents from {directory} ===")

        exact = ApproximateRetriever(path, exact=True)
        exact.index(answer_ids, [])
        exact_results, exact_latencies = time_search(exact, query_vectors, args.k, args.warmup)
        runs.append(
            {
                "documents": documents,
                "artifact": str(directory.resolve()),
                "index": "exact",
                "ef_search": None,
                "recall_at_k": 1.0,
                **summarise(exact_latencies),
            }
        )
        print(f"exact      p50 {runs[-1]['p50_ms']:>7.3f}ms  recall 1.000")
        write_result(report, args.output)

        approximate = ApproximateRetriever(
            path, neighbours=args.neighbours, ef_construction=args.ef_construction
        )
        build_started = time.perf_counter()
        approximate.index(answer_ids, [])
        build_seconds = time.perf_counter() - build_started

        for ef_search in args.ef_search:
            approximate.set_ef_search(ef_search)

            results, latencies = time_search(approximate, query_vectors, args.k, args.warmup)
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
                    "artifact": str(directory.resolve()),
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
            write_result(report, args.output)

    report["complete"] = True
    report["summary"] = best_settings(runs, args.recall_target)
    write_result(report, args.output)
    print(f"\nwrote {args.output}")

    # The headline: at each size, the fastest setting that keeps recall high.
    print(f"\nfastest HNSW setting reaching {args.recall_target} recall, against exact:")
    for summary in report["summary"]:
        documents, exact_p50 = summary["documents"], summary["exact_p50_ms"]
        print(summary["artifact"])
        best = summary["best"]
        if best is None:
            print(f"{documents:>8} documents: no eligible setting, exact {exact_p50:.2f}ms")
            continue
        verdict = "HNSW wins" if best["p50_ms"] < exact_p50 else "exact still wins"
        print(
            f"{documents:>8} documents: ef={best['ef_search']:<4} "
            f"{best['p50_ms']:.2f}ms against exact {exact_p50:.2f}ms   {verdict}"
        )


if __name__ == "__main__":
    main()
