"""Mine hard negatives for the training split.

    python scripts/mine_negatives.py --config configs/negatives.yaml

Writes data/processed/negatives.parquet. Mining is separate from training so a
retrain does not pay for it again, and so the negatives a run used are a file
that can be inspected rather than a side effect.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant_retrieval.models.negatives import (  # noqa: E402
    NegativeMiningConfig,
    combine_negatives,
    mine_negatives,
    sample_random_negatives,
)
from quant_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from quant_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from quant_retrieval.runtime import set_seed  # noqa: E402


def build_retriever(spec: dict):
    name = spec["retriever"]
    parameters = dict(spec.get("parameters", {}))
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
    set_seed(int(config.get("seed", 17)))
    mining = NegativeMiningConfig(**config.get("mining", {}))

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    qrels = pd.read_parquet(args.data / "qrels.parquet")
    training_queries = queries[queries["split"] == config.get("split", "train")]

    tables = []
    for spec in config["sources"]:
        table = mine_negatives(
            build_retriever(spec),
            corpus,
            training_queries,
            qrels,
            config=mining,
            source=spec["name"],
        )
        print(f"{spec['name']}: {len(table)} negatives")
        tables.append(table)

    random_per_query = int(config.get("random_per_query", 0))
    if random_per_query:
        drawn = sample_random_negatives(
            corpus,
            training_queries,
            qrels,
            per_query=random_per_query,
            seed=int(config.get("seed", 17)),
        )
        print(f"random: {len(drawn)} negatives")
        tables.append(drawn)

    negatives = combine_negatives(*tables)
    output = Path(config.get("output", args.data / "negatives.parquet"))
    output.parent.mkdir(parents=True, exist_ok=True)
    negatives.to_parquet(output, index=False)

    per_query = negatives.groupby("question_id").size()
    summary = {
        "negatives": int(len(negatives)),
        "questions_covered": int(len(per_query)),
        "training_queries": int(len(training_queries)),
        "per_query_min": int(per_query.min()),
        "per_query_median": int(per_query.median()),
        "per_query_max": int(per_query.max()),
        "by_source": negatives["source"].value_counts().to_dict(),
    }
    print(f"wrote {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
