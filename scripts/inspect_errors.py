"""Summarize a saved validation run without loading or running a model."""

import argparse
import hashlib
from pathlib import Path

from quant_retrieval.eval.analysis import error_report
from quant_retrieval.eval.metrics import METRIC_NAMES
from quant_retrieval.eval.results import parse_record, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metric", choices=METRIC_NAMES, default="ndcg_at_10")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.run.resolve():
        parser.error("analysis output must not replace its source evaluation")
    payload = args.run.read_bytes()
    report = error_report(parse_record(payload, args.run), args.metric, args.limit)
    report["source_sha256"] = hashlib.sha256(payload).hexdigest()
    write_result(report, args.output)
    print(f"wrote {args.output}: {report['queries']} queries, {report['status_counts']}")


if __name__ == "__main__":
    main()
