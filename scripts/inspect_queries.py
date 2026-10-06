"""Inspect a portable query cache without loading a model or search runtime."""

import argparse
from pathlib import Path

from quant_retrieval.eval.results import write_result
from quant_retrieval.retrieval.index_artifacts import file_digest
from quant_retrieval.retrieval.query_artifacts import QUERY_VECTOR_FILE, load_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="saved query directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.queries.resolve()):
        parser.error("inspection output must be outside the query artifact")
    if args.output.exists() or args.output.is_symlink():
        parser.error("inspection output exists; choose a new path")
    manifest, vectors = load_queries(args.queries)
    report = {"scope": "query_cache_integrity_only", "manifest": manifest,
              "source": str(args.queries.resolve()),
              "manifest_sha256": file_digest(args.queries / "manifest.json"),
              "payload_checksum_verified": True, "normalization_verified": True,
              "array_bytes": vectors.nbytes,
              "file_bytes": (args.queries / QUERY_VECTOR_FILE).stat().st_size}
    write_result(report, args.output, overwrite=False)
    print(f"verified {len(vectors)} saved queries; wrote {args.output}")


if __name__ == "__main__":
    main()
