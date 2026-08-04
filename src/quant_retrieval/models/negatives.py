"""Mine hard negatives: wrong answers that a retriever already ranks highly.

In-batch negatives are free but they get boring fast. Once the model can tell a
swaption question from a random answer about backtesting, the other 63 documents
in a batch stop teaching it anything, and the gradient goes quiet. Hard negatives
are the fix: ask a retriever for the answers it thinks belong to a question, drop
the ones that actually do, and what is left is wrong in a way the model currently
finds convincing.

The part that has to be right is what counts as wrong. Every answer written for
the same question is excluded, not only the judged positive. A grade 1 sibling is
a real answer to that question, and a downvoted answer is at least on topic, so
training the model to push either one away teaches it something false. False
negatives are worse than no negatives, because the model believes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from quant_retrieval.retrieval.base import Retriever


@dataclass(frozen=True)
class NegativeMiningConfig:
    # How deep to look. Anything the retriever puts this far down is not a
    # convincing wrong answer, it is just a document.
    depth: int = 50
    # How many to keep per question.
    per_query: int = 4
    # Skip the very top hits. The rank 1 miss for a question with no accepted
    # answer is often a decent answer that nobody voted on, and this corpus has
    # plenty of those. Starting a little lower trades a bit of hardness for
    # fewer mislabelled pairs.
    skip_top: int = 1


def mine_negatives(
    retriever: Retriever,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    config: NegativeMiningConfig | None = None,
    source: str = "unknown",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Return a table of (question_id, answer_id, rank, source) negatives."""
    config = config or NegativeMiningConfig()
    if config.per_query < 1:
        raise ValueError("per_query must be at least 1")
    if config.depth <= config.skip_top:
        raise ValueError("depth must be greater than skip_top")

    answer_to_question = dict(
        zip(corpus["answer_id"].astype(int), corpus["question_id"].astype(int), strict=True)
    )
    judged: dict[int, set[int]] = {}
    for row in qrels.itertuples(index=False):
        judged.setdefault(int(row.question_id), set()).add(int(row.answer_id))

    retriever.index(corpus["answer_id"].astype(int).tolist(), corpus["text"].tolist())

    rows: list[dict] = []
    for query in tqdm(
        queries.itertuples(index=False), total=len(queries), disable=not show_progress,
        desc=f"mining {source}",
    ):
        question_id = int(query.question_id)
        excluded = judged.get(question_id, set())
        kept = 0
        for rank, result in enumerate(retriever.search(query.text, config.depth)):
            if rank < config.skip_top:
                continue
            answer_id = int(result.document_id)
            if answer_id in excluded:
                continue
            # Any answer to this same question is off limits, judged or not.
            if answer_to_question.get(answer_id) == question_id:
                continue
            rows.append(
                {
                    "question_id": question_id,
                    "answer_id": answer_id,
                    "rank": rank,
                    "source": source,
                }
            )
            kept += 1
            if kept >= config.per_query:
                break

    return pd.DataFrame(rows, columns=["question_id", "answer_id", "rank", "source"])


def combine_negatives(*tables: pd.DataFrame) -> pd.DataFrame:
    """Merge negatives from several retrievers, keeping the best rank for each pair.

    Mining with two different retrievers finds different mistakes, and a document
    both of them rank highly is the most useful kind. Deduplicating on the pair
    rather than concatenating keeps one row per (question, answer).
    """
    if not tables:
        raise ValueError("nothing to combine")
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.sort_values(["question_id", "rank"])
    return combined.drop_duplicates(["question_id", "answer_id"]).reset_index(drop=True)
