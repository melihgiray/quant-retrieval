"""Check a reranker against documents drawn at random.

    python scripts/probe_reranker.py --checkpoint checkpoints/reranker_mixed/epoch-2

Two minutes, and it answers a question a full pipeline evaluation takes twenty
minutes to answer badly: can this model tell the right answer from documents that
are not even about the topic. A reranker that cannot do that will destroy any
ranking it is given, and no amount of further training on hard negatives will
fix it.

Run it on the training split as well as validation. A model that fails here on
questions it trained on is not undertrained, it was trained on the wrong
distribution, which is exactly what happened to the first reranker in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from quant_retrieval.data.pairs import GRADE_PRIMARY  # noqa: E402
from quant_retrieval.models.cross_encoder import CrossEncoder  # noqa: E402
from quant_retrieval.runtime import choose_device, set_seed  # noqa: E402


def probe_split(
    model,
    tokenizer,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    split: str,
    questions: int,
    distractors: int,
    max_length: int,
    device: str,
    seed: int,
) -> dict:
    """Share of questions whose own answer outscores N random documents."""
    texts = corpus.set_index("answer_id")["text"]
    gold = qrels.loc[qrels["grade"] == GRADE_PRIMARY].set_index("question_id")["answer_id"]
    selected = queries.loc[queries["split"] == split]
    selected = selected[selected["question_id"].isin(gold.index)].head(questions)
    generator = np.random.default_rng(seed)
    answer_ids = corpus["answer_id"].to_numpy()

    wins = 0
    margins: list[float] = []
    for row in selected.itertuples(index=False):
        positive = texts.loc[gold.loc[row.question_id]]
        drawn = texts.loc[generator.choice(answer_ids, distractors, replace=False)].tolist()
        candidates = [positive, *drawn]
        tokens = tokenizer(
            [row.text] * len(candidates),
            candidates,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        with torch.inference_mode():
            scores = model(**tokens).cpu().numpy()
        wins += int(scores.argmax() == 0)
        margins.append(float(scores[0] - scores[1:].max()))

    total = len(selected)
    return {
        "split": split,
        "questions": total,
        "distractors": distractors,
        "top_one_accuracy": round(wins / total, 4),
        "chance": round(1 / (distractors + 1), 4),
        "median_margin": round(float(np.median(margins)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device("auto")
    model = CrossEncoder.load(args.checkpoint).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))

    corpus = pd.read_parquet(args.data / "corpus.parquet")
    queries = pd.read_parquet(args.data / "queries.parquet")
    qrels = pd.read_parquet(args.data / "qrels.parquet")

    report = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "splits": [
            probe_split(
                model,
                tokenizer,
                corpus,
                queries,
                qrels,
                split=split,
                questions=args.questions,
                distractors=args.distractors,
                max_length=args.max_length,
                device=device,
                seed=args.seed,
            )
            for split in ("val", "train")
        ],
    }

    output = args.output or Path("results") / f"{args.checkpoint.parent.name}_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output}")
    for entry in report["splits"]:
        print(
            f"{entry['split']:>5}: {entry['top_one_accuracy']:.1%} correct against "
            f"{entry['distractors']} random documents, chance {entry['chance']:.1%}"
        )
    print()
    print(
        "A working reranker should be far above chance on both, and the training "
        "split should not be worse than validation."
    )


if __name__ == "__main__":
    main()
