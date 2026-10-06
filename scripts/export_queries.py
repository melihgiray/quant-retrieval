"""Encode validation queries once in a process that never imports FAISS."""

import argparse
import os
from pathlib import Path

import pandas as pd

from quant_retrieval.eval.benchmark import benchmark_context
from quant_retrieval.eval.sampling import sample_queries
from quant_retrieval.retrieval.checkpoint import verify_checkpoint
from quant_retrieval.retrieval.index_artifacts import (
    file_digest,
    read_manifest,
    verify_export_files,
)
from quant_retrieval.retrieval.query_artifacts import publish_queries, validate_query_manifest
from quant_retrieval.retrieval.runtime_isolation import ensure_encoder_runtime


def encode_queries(
    checkpoint: Path, texts: list[str], metadata: dict, batch_size: int, device: str
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_encoder_runtime()
    from quant_retrieval.retrieval.dense import DenseRetriever
    from quant_retrieval.runtime import set_seed

    set_seed(metadata["seed"])
    encoder = DenseRetriever(str(checkpoint), batch_size=batch_size, device=device,
                             max_length=metadata["max_length"], pooling=metadata["pooling"],
                             show_progress=False)
    return encoder._encode(texts), encoder.device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="verified document export")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.queries, args.batch_size) <= 0 or not 0 <= args.seed < 2**32:
        parser.error("query and batch counts must be positive; seed must fit uint32")
    if args.output.exists() or args.output.is_symlink():
        parser.error("query output exists; choose a new version directory")
    if any(args.output.resolve().is_relative_to(path.resolve())
           for path in (args.checkpoint, args.index, args.data)):
        parser.error("query output must be outside input directories")
    path = args.index / "manifest.json"
    index = read_manifest(path)
    if not verify_checkpoint(args.checkpoint, index):
        parser.error("query export requires production-time checkpoint hashes")
    if not verify_export_files(args.index, index, ("answer_ids.npy", "embeddings_fp32.npy")):
        parser.error("query export requires a checksum-verified document index")
    selected = sample_queries(
        pd.read_parquet(args.data / "queries.parquet"), args.queries, args.seed
    )
    metadata = {**benchmark_context(selected, args.seed), "schema_version": 1,
                "kind": "ann_queries", "split": "val", "queries": len(selected),
                "dimensions": index.get("dimensions"), "max_length": index.get("max_length"),
                "pooling": index.get("pooling", "mean"), "dtype": "float32",
                "checkpoint_sha256": index["checkpoint_sha256"], "vectors_sha256": "0" * 64,
                "source_manifest_sha256": file_digest(path), "batch_size": args.batch_size}
    validate_query_manifest(metadata)
    vectors, device = encode_queries(args.checkpoint, selected.text.tolist(), metadata,
                                    args.batch_size, args.device)
    if vectors.shape != (len(selected), metadata["dimensions"]):
        raise ValueError("query encoder dimensions disagree with document export")
    verify_checkpoint(args.checkpoint, index)
    publish_queries(args.output, vectors, {**metadata, "device": device})
    print(f"saved {len(selected)} validation query vectors to {args.output}")


if __name__ == "__main__":
    main()
