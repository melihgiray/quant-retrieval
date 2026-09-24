"""Summarize a saved validation run without loading or running a model."""

import argparse
import hashlib
import json
from pathlib import Path

from quant_retrieval.eval.analysis import error_report
from quant_retrieval.eval.metrics import METRIC_NAMES
from quant_retrieval.eval.results import write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metric", choices=METRIC_NAMES, default="ndcg_at_10")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.run.read_bytes()
    report = error_report(json.loads(payload), args.metric, args.limit)
    report["source_sha256"] = hashlib.sha256(payload).hexdigest()
    write_result(report, args.output)
    print(f"wrote {args.output}: {report['queries']} queries, {report['status_counts']}")


if __name__ == "__main__":
    main()
