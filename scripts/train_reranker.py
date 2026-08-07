"""Fine-tune the cross-encoder reranker.

    python scripts/train_reranker.py --config configs/reranker.yaml

Needs mined negatives, since the reranker learns by separating the right answer
from candidates that already look right. Run scripts/mine_negatives.py first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant_retrieval.models.dataset import load_training_pairs  # noqa: E402
from quant_retrieval.models.train import TrainingConfig, write_history  # noqa: E402
from quant_retrieval.models.train_reranker import train_reranker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    run_name = raw.pop("run_name")
    history_path = Path(raw.pop("history", f"results/{run_name}_training.json"))
    config = TrainingConfig(output_dir=Path(raw.pop("output_dir")), **raw)

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    qrels = pd.read_parquet(args.data / "qrels.parquet")
    negatives = pd.read_parquet(config.negatives_path)
    pairs = load_training_pairs(
        corpus,
        queries,
        qrels,
        split="train",
        negatives=negatives,
        negatives_per_query=config.negatives_per_query,
    )

    history = train_reranker(config, pairs, resume=args.resume)
    write_history(history, history_path, config)

    first, last = history.epochs[0], history.epochs[-1]
    print(f"wrote {history_path}")
    print(
        f"mean loss {first['mean_loss']:.4f} -> {last['mean_loss']:.4f}, "
        f"top-1 {first['mean_top_one_accuracy']:.3f} -> {last['mean_top_one_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
