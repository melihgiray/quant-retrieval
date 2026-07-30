"""Evaluate one retrieval configuration against the full answer corpus.

    python scripts/evaluate.py --config configs/bm25.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.eval.results import build_result_record, write_result
from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever


def build_retriever(config: dict[str, Any]):
    name = config["retriever"]
    parameters = config.get("parameters", {})
    if name == "bm25":
        return BM25Retriever(**parameters)
    if name == "dense":
        return DenseRetriever(**parameters)
    raise ValueError(f"unknown retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
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
    )
    output = Path(config.get("output", f"results/{config['run_name']}.json"))
    write_result(record, output)

    print(f"wrote {output}")
    for name, value in record["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
