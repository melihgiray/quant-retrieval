"""Fine-tune the retrieval encoder on the training split.

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/base.yaml

Writes one Hugging Face checkpoint per epoch under the configured output
directory, plus the loss history to results/<run_name>_training.json. Evaluate a
checkpoint by pointing a dense retriever config at its directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before transformers imports a tokenizer. The fast tokenizers fork under
# the hood and warn, or worse deadlock, when a DataLoader forks around them.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant_retrieval.models.dataset import load_training_pairs  # noqa: E402
from quant_retrieval.models.train import TrainingConfig, train, write_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--resume", action="store_true", help="continue from the last completed epoch"
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    run_name = raw.pop("run_name")
    history_path = Path(raw.pop("history", f"results/{run_name}_training.json"))
    config = TrainingConfig(output_dir=Path(raw.pop("output_dir")), **raw)

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    qrels = pd.read_parquet(args.data / "qrels.parquet")
    pairs = load_training_pairs(corpus, queries, qrels, split="train")

    history = train(config, pairs, resume=args.resume)
    write_history(history, history_path, config)

    first = history.epochs[0]
    last = history.epochs[-1]
    print(f"wrote {history_path}")
    print(
        f"mean loss {first['mean_loss']:.4f} -> {last['mean_loss']:.4f}, "
        f"in-batch accuracy {first['mean_in_batch_accuracy']:.3f} -> "
        f"{last['mean_in_batch_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
