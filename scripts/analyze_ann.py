"""Audit a saved ANN sweep and compare operating points without loading a model.

    python -m scripts.analyze_ann --report results/ann_scaling.json \
        --output results/ann_analysis.json --recall-target 0.95
"""

import argparse
import hashlib
from pathlib import Path

from quant_retrieval.eval.ann_analysis import analyze_ann_report
from quant_retrieval.eval.results import parse_record, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-target", type=float)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error("analysis output must differ from the measurement report")
    if args.output.exists() or args.output.is_symlink():
        parser.error("analysis output exists; choose a new path")
    contents = args.report.read_bytes()
    report = parse_record(contents, args.report)
    try:
        result = analyze_ann_report(report, args.recall_target)
    except ValueError as error:
        parser.error(str(error))
    result["source"] = {
        "path": str(args.report.resolve()), "sha256": hashlib.sha256(contents).hexdigest(),
        "benchmark": {key: report.get(key) for key in (
            "commit", "created_at", "environment", "seed", "query_sha256", "question_ids",
            "queries", "repeats", "warmup", "threads", "k", "neighbours", "ef_construction",
        )},
    }
    write_result(result, args.output, overwrite=False)
    print(f"audited {len(result['artifacts'])} artifacts; wrote {args.output}")


if __name__ == "__main__":
    main()
