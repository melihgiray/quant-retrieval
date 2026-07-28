"""Build the retrieval dataset from the raw dump.

    python scripts/build_dataset.py

Reads data/raw/*.xml, writes data/processed/{corpus,queries,qrels}.parquet and
results/dataset_stats.json. The stats file is what docs/DATA.md quotes, so the
numbers in the docs always come from a run rather than from memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_retrieval.data.pairs import GRADE_PRIMARY, build_corpus, build_qrels, build_queries
from quant_retrieval.data.parse import parse_post_links, parse_posts
from quant_retrieval.data.splits import (
    assign_time_splits,
    drop_cross_split_duplicates,
    split_boundaries,
)

HELD_OUT = ("val", "test")


def text_stats(series: pd.Series) -> dict[str, int]:
    words = series.str.split().str.len()
    return {
        "count": int(len(series)),
        "chars_p50": int(series.str.len().quantile(0.5)),
        "chars_p90": int(series.str.len().quantile(0.9)),
        "words_p50": int(words.quantile(0.5)),
        "words_p90": int(words.quantile(0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--stats", type=Path, default=Path("results/dataset_stats.json"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    questions, answers = parse_posts(args.raw / "Posts.xml")
    post_links = parse_post_links(args.raw / "PostLinks.xml")
    print(f"parsed {len(questions)} questions, {len(answers)} answers")

    corpus = build_corpus(answers)
    qrels = build_qrels(questions, answers)
    queries = build_queries(questions)

    # A question is only usable as a query if something judges it.
    judged = set(qrels.loc[qrels["grade"] == GRADE_PRIMARY, "question_id"])
    queries = queries[queries["question_id"].isin(judged)].reset_index(drop=True)
    print(f"{len(queries)} questions carry a primary judgement")

    queries = assign_time_splits(queries, val_frac=args.val_frac, test_frac=args.test_frac)
    queries, dropped = drop_cross_split_duplicates(queries, post_links)
    print(f"dropped {len(dropped)} questions that duplicate a higher priority split")

    # Validation and test are scored on accepted answers only. The top voted
    # stand in is good enough to learn from and too noisy to report on.
    primary = qrels[qrels["grade"] == GRADE_PRIMARY].set_index("question_id")["label_source"]
    queries["label_source"] = queries["question_id"].map(primary)
    noisy_held_out = queries["split"].isin(HELD_OUT) & (queries["label_source"] == "top_voted")
    print(f"dropped {int(noisy_held_out.sum())} held out questions with no accepted answer")
    queries = queries[~noisy_held_out].reset_index(drop=True)

    qrels = qrels[qrels["question_id"].isin(set(queries["question_id"]))].reset_index(drop=True)

    check_invariants(queries, qrels, corpus)

    args.out.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(args.out / "corpus.parquet", index=False)
    queries.to_parquet(args.out / "queries.parquet", index=False)
    qrels.to_parquet(args.out / "qrels.parquet", index=False)

    stats = {
        "raw": json.loads((args.raw / "dump_info.json").read_text()),
        "posts": {"questions": int(len(questions)), "answers": int(len(answers))},
        "corpus": text_stats(corpus["text"]),
        "queries_by_split": {
            split: text_stats(group["text"])
            for split, group in queries.groupby("split", sort=False)
        },
        "split_boundaries": split_boundaries(queries),
        "label_sources": queries["label_source"].value_counts().to_dict(),
        "qrels_by_grade": qrels["grade"].value_counts().to_dict(),
        "judgements_per_query": round(len(qrels) / len(queries), 2),
        "dropped_as_cross_split_duplicates": int(len(dropped)),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2, default=str) + "\n")

    print(json.dumps(stats, indent=2, default=str))


def check_invariants(queries: pd.DataFrame, qrels: pd.DataFrame, corpus: pd.DataFrame) -> None:
    """Fail the build rather than train on a broken dataset."""
    by_split = queries.groupby("split")["question_id"].apply(set)
    train, test = by_split.get("train", set()), by_split.get("test", set())
    val = by_split.get("val", set())
    assert train.isdisjoint(test), "train and test share questions"
    assert train.isdisjoint(val), "train and val share questions"
    assert val.isdisjoint(test), "val and test share questions"

    judged_answers = set(qrels["answer_id"])
    assert judged_answers <= set(corpus["answer_id"]), "a judged answer is missing from the corpus"

    primary = qrels[qrels["grade"] == GRADE_PRIMARY]
    assert primary["question_id"].is_unique, "a question has two primary judgements"
    assert set(queries["question_id"]) == set(primary["question_id"]), (
        "every query needs exactly one primary judgement"
    )

    held_out = queries[queries["split"].isin(HELD_OUT)]
    assert (held_out["label_source"] == "accepted").all(), (
        "held out query without an accepted answer"
    )


if __name__ == "__main__":
    main()
