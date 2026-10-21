"""Audit a saved train/validation evaluation without loading data or models."""

import argparse
import hashlib
from pathlib import Path

from quant_retrieval.eval.audit import audit_result
from quant_retrieval.eval.results import parse_record, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.run.resolve():
        parser.error("audit output must differ from its source")
    if args.output.exists() or args.output.is_symlink():
        parser.error("audit output exists; choose a new path")
    contents = args.run.read_bytes()
    try:
        report = audit_result(parse_record(contents, args.run))
    except ValueError as error:
        parser.error(str(error))
    report["source"] = {
        "path": str(args.run.resolve()),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }
    write_result(report, args.output, overwrite=False)
    print(f"audited {report['protocol']['counts']['queries']} queries; wrote {args.output}")


if __name__ == "__main__":
    main()
