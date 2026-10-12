"""Check snapshot consistency offline and save a non-overwriting JSON report."""

import argparse
from pathlib import Path

from quant_retrieval.eval.results import write_result
from quant_retrieval.serve.snapshot_validation import inspect_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("inspection output already exists")
    if args.output.resolve().is_relative_to(args.snapshot.resolve()):
        raise ValueError("inspection output must be outside the snapshot")
    write_result(inspect_snapshot(args.snapshot), args.output, overwrite=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
