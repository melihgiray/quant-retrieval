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
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quant_retrieval.eval.benchmark import benchmark_context, runtime_context  # noqa: E402
from quant_retrieval.eval.results import write_result  # noqa: E402
from quant_retrieval.eval.sampling import sample_queries  # noqa: E402
from quant_retrieval.retrieval.ann import ApproximateRetriever, recall_against_exact  # noqa: E402
from quant_retrieval.retrieval.checkpoint import verify_checkpoint  # noqa: E402
from quant_retrieval.retrieval.index_artifacts import (  # noqa: E402
    file_digest,
    read_manifest,
    validate_document_ids,
    verify_export_files,
)
from quant_retrieval.retrieval.query_artifacts import (  # noqa: E402
    load_queries,
    verify_query_compatibility,
)
from quant_retrieval.retrieval.runtime_isolation import ensure_ann_runtime  # noqa: E402
from quant_retrieval.retrieval.vectors import validate_embeddings  # noqa: E402

DEFAULT_EF_SEARCH = (16, 32, 64, 128, 256)


def encode_live_queries(checkpoint: Path, selected: pd.DataFrame, manifest: dict, seed: int):
    from quant_retrieval.retrieval.dense import DenseRetriever
    from quant_retrieval.runtime import set_seed

    set_seed(seed)
    encoder = DenseRetriever(str(checkpoint), max_length=manifest["max_length"],
                             show_progress=False, pooling=manifest.get("pooling", "mean"))
    vectors = encoder._encode(selected["text"].tolist())
    print(f"encoded {len(vectors)} queries on {encoder.device}")
    return vectors


def load_manifest(directory: Path, checkpoint: Path | None, query_manifest: dict | None = None):
    manifest = read_manifest(directory / "manifest.json")
    for key in ("documents", "dimensions", "max_length"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError(f"{directory}: {key} must be a positive integer")
    if not isinstance(manifest.get("checkpoint"), str) or not manifest["checkpoint"]:
        raise ValueError(f"{directory}: checkpoint is required")
    if query_manifest is not None:
        verify_query_compatibility(query_manifest, manifest)
    elif checkpoint is not None:
        verified = verify_checkpoint(checkpoint, manifest)
        if not verified and Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"{directory}: checkpoint does not match the query encoder")
    else:
        raise ValueError("a query checkpoint or verified query cache is required")
    if manifest.get("pooling", "mean") not in {"mean", "cls"}:
        raise ValueError(f"{directory}: unsupported pooling strategy")
    verified_files = verify_export_files(directory, manifest,
                                         ("answer_ids.npy", "embeddings_fp32.npy"))
    if query_manifest is not None and not verified_files:
        raise ValueError("cached-query sweeps require checksum-verified document exports")
    ids = np.load(directory / "answer_ids.npy", mmap_mode="r")
    vectors = np.load(directory / "embeddings_fp32.npy", mmap_mode="r")
    if ids.shape != (manifest["documents"],):
        raise ValueError(f"{directory}: document count disagrees with answer IDs")
    if vectors.shape != (manifest["documents"], manifest["dimensions"]):
        raise ValueError(f"{directory}: vector shape disagrees with manifest")
    validate_document_ids(ids)
    if vectors.dtype != np.float32:
        raise ValueError(f"{directory}: ANN exports must use float32 vectors")
    validate_embeddings(vectors, len(ids), atol=1e-4)
    return manifest


def time_search(
    retriever, queries: np.ndarray, k: int, warmup: int = 0, repeats: int = 1
) -> tuple[list, list[float]]:
    """Keep one ranking per query and latency samples from every measured pass."""
    if (type(warmup) is not int or type(repeats) is not int or type(k) is not int
            or warmup < 0 or repeats <= 0 or k <= 0
            or not isinstance(queries, np.ndarray) or queries.ndim != 2 or not len(queries)):
        raise ValueError("warmup must be nonnegative; queries and repeats must be positive")
    for index in range(warmup):
        retriever.search_vector(queries[index % len(queries)], k)
    results, latencies = [], []
    for repetition in range(repeats):
        for vector in queries:
            started = time.perf_counter()
            ranking = retriever.search_vector(vector, k)
            latencies.append((time.perf_counter() - started) * 1000)
            if repetition == 0:
                results.append(ranking)
    return results, latencies


def summarise(latencies: list[float]) -> dict[str, float | int]:
    values = np.asarray(latencies)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("latencies must be nonempty finite nonnegative numbers")
    return {
        "samples": len(latencies),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
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
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--queries", type=int)
    parser.add_argument("--query-cache", type=Path, help="saved vectors; no model runtime needed")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ef-search", nargs="+", type=int, default=list(DEFAULT_EF_SEARCH))
    parser.add_argument("--neighbours", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1, help="FAISS CPU threads")
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/ann_scaling.json"))
    parser.add_argument("--overwrite", action="store_true", help="replace an earlier report")
    args = parser.parse_args()
    if args.query_cache:
        live_options = (args.data, args.checkpoint, args.queries, args.seed)
        if any(value is not None for value in live_options):
            parser.error("query-cache cannot be combined with data, checkpoint, queries or seed")
    else:
        args.data = args.data or Path("data/processed")
        args.checkpoint = args.checkpoint or Path("checkpoints/minilm_tuned/epoch-3")
        args.queries = 200 if args.queries is None else args.queries
        args.seed = 17 if args.seed is None else args.seed
    inputs = [args.data / "queries.parquet"] if args.data else []
    for directory in args.embeddings:
        inputs.extend(directory / name for name in (
            "manifest.json", "answer_ids.npy", "embeddings_fp32.npy", "embeddings_fp16.npy"
        ))
    if (args.output.resolve() in {path.resolve() for path in inputs}
            or any(args.output.resolve().is_relative_to(root.resolve())
                   for root in (args.checkpoint, args.query_cache) if root is not None)):
        parser.error("output must not replace benchmark inputs or checkpoint files")
    if (args.output.exists() or args.output.is_symlink()) and not args.overwrite:
        parser.error("output exists; choose a new path or pass --overwrite")
    if any(value <= 0 for value in [args.k, args.neighbours, args.repeats,
                                    args.ef_construction, args.threads, *args.ef_search]
           + ([] if args.queries is None else [args.queries])):
        parser.error("query counts, graph settings and threads must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not 0 <= args.recall_target <= 1:
        parser.error("recall target must lie between zero and one")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("seed must fit uint32")
    if len(set(args.ef_search)) != len(args.ef_search):
        parser.error("ef-search settings must be unique")
    if len({path.resolve() for path in args.embeddings}) != len(args.embeddings):
        parser.error("embedding directories must be unique")
    try:
        ensure_ann_runtime(args.query_cache is not None)
    except ValueError as error:
        parser.error(str(error))

    query_manifest = None
    if args.query_cache:
        query_manifest, query_vectors = load_queries(args.query_cache)
    manifests = [load_manifest(directory, args.checkpoint, query_manifest)
                 for directory in args.embeddings]
    if len({(m["dimensions"], m["max_length"], m.get("pooling", "mean"))
            for m in manifests}) != 1:
        parser.error("all embedding sets must use the same dimensions, max_length and pooling")

    import faiss

    faiss.omp_set_num_threads(args.threads)

    if query_manifest is None:
        queries = pd.read_parquet(args.data / "queries.parquet")
        selected = sample_queries(queries, args.queries, args.seed)
        query_vectors = encode_live_queries(args.checkpoint, selected, manifests[0], args.seed)
        validate_embeddings(query_vectors, len(selected), atol=1e-4)
        if query_vectors.shape != (len(selected), manifests[0]["dimensions"]):
            raise ValueError("query encoder dimensions disagree with exported vectors")
        context = benchmark_context(selected, args.seed)
        query_source = {"kind": "live"}
    else:
        context = {**runtime_context(), **{key: query_manifest[key]
                   for key in ("seed", "question_ids", "query_sha256")}}
        query_source = {"kind": "cache", "directory": str(args.query_cache.resolve()),
                        "manifest_sha256": file_digest(args.query_cache / "manifest.json"),
                        "manifest": query_manifest}

    runs = []
    report = {
        **context,
        "query_source": query_source,
        "ann_runtime": {"faiss": getattr(faiss, "__version__", None), "numpy": np.__version__},
        "schema_version": 1,
        "scope": "index_search_only",
        "latency_order": "repeat_major_query_minor",
        "recall_reference": "exact_top_k_overlap",
        "artifacts": [
            {"directory": str(directory.resolve()), "manifest": manifest,
             "payload_checksums_verified": manifest.get("sha256") is not None,
             "checkpoint_checksums_verified": "checkpoint_sha256" in manifest}
            for directory, manifest in zip(args.embeddings, manifests, strict=True)
        ],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "k": args.k, "queries": len(query_vectors), "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads, "neighbours": args.neighbours,
        "ef_construction": args.ef_construction, "ef_search": args.ef_search,
        "complete": False, "runs": runs,
        "recall_target": args.recall_target,
    }
    write_result(report, args.output, overwrite=args.overwrite)
    for directory, manifest in zip(args.embeddings, manifests, strict=True):
        answer_ids = np.load(directory / "answer_ids.npy").tolist()
        path = directory / "embeddings_fp32.npy"
        documents = manifest["documents"]
        print(f"\n=== {documents} documents from {directory} ===")

        exact = ApproximateRetriever(path, exact=True)
        exact_started = time.perf_counter()
        exact.index(answer_ids, [])
        exact_build_seconds = time.perf_counter() - exact_started
        exact_results, exact_latencies = time_search(
            exact, query_vectors, args.k, args.warmup, args.repeats
        )
        runs.append(
            {
                "documents": documents,
                "artifact": str(directory.resolve()),
                "index": "exact",
                "build_seconds": exact_build_seconds,
                "ef_search": None,
                "recall_at_k": 1.0,
                "per_query_recall": [1.0] * len(query_vectors),
                "latencies_ms": exact_latencies,
                **summarise(exact_latencies),
            }
        )
        print(f"exact      p50 {runs[-1]['p50_ms']:>7.3f}ms  recall 1.000")
        write_result(report, args.output)

        # Keep rankings, not a second full vector index, while building the graph.
        del exact
        approximate = ApproximateRetriever(
            path, neighbours=args.neighbours, ef_construction=args.ef_construction
        )
        build_started = time.perf_counter()
        approximate.index(answer_ids, [])
        build_seconds = time.perf_counter() - build_started

        for ef_search in args.ef_search:
            approximate.set_ef_search(ef_search)

            results, latencies = time_search(
                approximate, query_vectors, args.k, args.warmup, args.repeats
            )
            per_query_recall = [
                recall_against_exact(got, want, args.k)
                for got, want in zip(results, exact_results, strict=True)
            ]
            recall = float(np.mean(per_query_recall))
            runs.append(
                {
                    "documents": documents,
                    "artifact": str(directory.resolve()),
                    "index": "hnsw",
                    "ef_search": ef_search,
                    "neighbours": args.neighbours,
                    "recall_at_k": recall,
                    "per_query_recall": per_query_recall,
                    "latencies_ms": latencies,
                    "build_seconds": build_seconds,
                    **summarise(latencies),
                }
            )
            print(
                f"hnsw ef={ef_search:<4} p50 {runs[-1]['p50_ms']:>7.3f}ms  "
                f"recall {recall:.3f}  build {build_seconds:.0f}s"
            )
            write_result(report, args.output)
        del approximate

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
