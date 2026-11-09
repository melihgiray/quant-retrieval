"""Summarize audited validation scores by diagnostic or verified query-text groups."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from quant_retrieval.eval.results import parse_record, write_result
from quant_retrieval.eval.slices import (
    compare_groups,
    diagnostic_labels,
    grouped_summary,
    query_length_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="compare against the fixed baseline groups")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-by",
        required=True,
        choices=("diagnostic_status", "primary_labels", "first_primary_rank", "query_length"),
    )
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--word-boundaries", type=int, nargs="+")
    parser.add_argument("--min-queries", type=int, default=5)
    args = parser.parse_args()
    if args.group_by == "query_length" and args.queries is None:
        parser.error("query_length requires --queries")
    if args.group_by != "query_length" and (args.queries or args.word_boundaries is not None):
        parser.error("query-table options require query_length grouping")
    sources = [
        path.resolve() for path in (args.run, args.queries, args.candidate) if path is not None
    ]
    if args.output.resolve() in sources:
        parser.error("slice output must not replace an input")
    if args.output.exists() or args.output.is_symlink():
        parser.error("slice output exists; choose a new path")
    contents = args.run.read_bytes()
    record = parse_record(contents, args.run)
    definition = {
        "group_by": args.group_by,
        "assignment_source": "baseline" if args.candidate else "source_run",
    }
    source = {"path": str(args.run.resolve()), "sha256": hashlib.sha256(contents).hexdigest()}
    try:
        if args.group_by == "query_length":
            boundaries = tuple(args.word_boundaries or (10, 25, 50, 100))
            labels = query_length_labels(record, pd.read_parquet(args.queries), boundaries)
            definition.update(word_boundaries=list(boundaries), length_unit="whitespace_words")
            source["query_table"] = {
                "path": str(args.queries.resolve()),
                "sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            }
        else:
            labels = diagnostic_labels(record, args.group_by)
        if args.candidate:
            candidate_bytes = args.candidate.read_bytes()
            candidate = parse_record(candidate_bytes, args.candidate)
            report = compare_groups(
                record, candidate, labels, group_by=args.group_by, min_queries=args.min_queries
            )
            source["candidate"] = {
                "path": str(args.candidate.resolve()),
                "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            }
        else:
            report = grouped_summary(
                record, labels, group_by=args.group_by, min_queries=args.min_queries
            )
    except ValueError as error:
        parser.error(str(error))
    report.update(source=source, grouping_definition=definition)
    write_result(report, args.output, overwrite=False)
    print(f"wrote {len(report['groups'])} query groups to {args.output}")


if __name__ == "__main__":
    main()
