"""Evaluate one retrieval configuration against the full answer corpus.

    python scripts/evaluate.py --config configs/bm25.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.eval.results import build_result_record, write_result
from quant_retrieval.retrieval.factory import build_retriever
from quant_retrieval.runtime import set_seed


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be an object")
    name = config.get("run_name")
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("run_name must be a filename-safe identifier")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between zero and 2**32 - 1")
    if config.get("split", "val") not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    count = config.get("max_results", 100)
    if type(count) is not int or count < 100:
        raise ValueError("max_results must be an integer of at least 100 for Recall@100")
    if config.get("retriever") not in ("bm25", "dense", "hybrid", "rerank"):
        raise ValueError("config must name a supported retriever")
    if not isinstance(config.get("parameters", {}), dict):
        raise ValueError("retriever parameters must be an object")
    if "output" in config and (
        not isinstance(config["output"], str) or not config["output"].strip()
    ):
        raise ValueError("output must be a nonempty path string")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--save-rankings", action="store_true",
                        help="include retrieved answer IDs for offline inspection")
    parser.add_argument("--allow-test", action="store_true",
                        help="explicitly authorize the final held-out test evaluation")
    parser.add_argument("--output", type=Path, help="override the configured result path")
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly replace an existing evaluation result")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    if config.get("split", "val") == "test" and not args.allow_test:
        parser.error("held-out test evaluation requires --allow-test; tune on val first")
    output = args.output or Path(config.get("output", f"results/{config['run_name']}.json"))
    inputs = [args.config, *(args.data / name for name in
                            ("corpus.parquet", "queries.parquet", "qrels.parquet"))]
    if output.resolve() in {path.resolve() for path in inputs}:
        parser.error("result output must not replace configuration or dataset inputs")
    if output.exists() and not args.overwrite:
        parser.error("result already exists; choose --output or explicitly pass --overwrite")
    set_seed(config["seed"])
    retriever = build_retriever(config)
    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    qrels = pd.read_parquet(args.data / "qrels.parquet")

    evaluation = evaluate_retriever(
        retriever,
        corpus,
        queries,
        qrels,
        split=config.get("split", "val"),
        max_results=config.get("max_results", 100),
    )
    record = build_result_record(
        run_name=config["run_name"],
        retriever=config["retriever"],
        split=config.get("split", "val"),
        config=config,
        evaluation=evaluation,
        include_rankings=args.save_rankings,
    )
    record["test_split_authorized"] = config.get("split", "val") == "test" and args.allow_test
    write_result(record, output, overwrite=args.overwrite)

    print(f"wrote {output}")
    for name, value in record["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
