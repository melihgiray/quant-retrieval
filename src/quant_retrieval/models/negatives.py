"""Wrong answers to train against, of two kinds.

Mined hard negatives come from asking a retriever which answers belong to a
question and dropping the ones that actually do. What is left is wrong in a way
the model currently finds convincing, which is what makes it worth learning from.
In-batch negatives get boring fast: once the model can tell a swaption question
from a random answer about backtesting, the rest of the batch teaches it nothing.

Random negatives are the other half, and the reranker is why they exist. Trained
on mined negatives alone it learned to make fine distinctions inside a narrow
band of plausible answers and never learned the coarse one, so at search time it
had no way to reject documents that were not even about the topic, which is most
of what it sees. Hard negatives sharpen a decision boundary. They cannot draw one
that was never there.

The part that has to be right for both is what counts as wrong. Every answer
written for the same question is excluded, not only the judged positive. A grade
1 sibling is a real answer, and a downvoted answer is at least on topic, so
training the model to push either away teaches it something false. False
negatives are worse than no negatives, because the model believes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from quant_retrieval.retrieval.base import Retriever

# Random draws rank above every mined hit, so combining the two keeps the
# retriever's rank whenever a document turns up in both.
RANDOM_RANK_OFFSET = 10_000


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


def build_exclusion_lookups(
    corpus: pd.DataFrame, qrels: pd.DataFrame
) -> tuple[dict[int, int], dict[int, set[int]]]:
    """What each question is not allowed to treat as a wrong answer.

    Returns the answer to question map and the judged answers per question. Both
    kinds of negative use these, and they have to agree: a rule that applies when
    mining and not when sampling would put a real answer in front of the model as
    a negative half the time.
    """
    answer_to_question = dict(
        zip(corpus["answer_id"].astype(int), corpus["question_id"].astype(int), strict=True)
    )
    judged: dict[int, set[int]] = {}
    for row in qrels.itertuples(index=False):
        judged.setdefault(int(row.question_id), set()).add(int(row.answer_id))
    return answer_to_question, judged


def sample_random_negatives(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    per_query: int = 2,
    seed: int = 17,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Draw wrong answers uniformly from the corpus.

    Mined negatives are convincing by construction, which is exactly why a model
    trained on nothing else never learns to reject a document that is not even
    about the topic. Most of what a reranker sees at search time is that easy
    case. These supply it.

    Ranks start high on purpose. `combine_negatives` keeps the lowest rank per
    pair, so a document that was both mined and drawn at random stays labelled
    with the rank the retriever gave it.
    """
    if per_query < 1:
        raise ValueError("per_query must be at least 1")

    answer_to_question, judged = build_exclusion_lookups(corpus, qrels)
    answer_ids = corpus["answer_id"].astype(int).to_numpy()
    if len(answer_ids) <= per_query:
        raise ValueError("corpus is too small to draw negatives from")
    generator = np.random.default_rng(seed)

    rows: list[dict] = []
    for query in tqdm(
        queries.itertuples(index=False),
        total=len(queries),
        disable=not show_progress,
        desc="sampling random",
    ):
        question_id = int(query.question_id)
        excluded = judged.get(question_id, set())
        kept: list[int] = []
        # Rejection sampling. The excluded set is a handful of documents out of
        # tens of thousands, so a draw almost never collides and the loop almost
        # never runs twice.
        while len(kept) < per_query:
            draw = generator.choice(answer_ids, size=per_query * 2, replace=False)
            for answer_id in (int(value) for value in draw):
                if answer_id in excluded or answer_to_question.get(answer_id) == question_id:
                    continue
                if answer_id in kept:
                    continue
                kept.append(answer_id)
                if len(kept) == per_query:
                    break

        rows.extend(
            {
                "question_id": question_id,
                "answer_id": answer_id,
                "rank": RANDOM_RANK_OFFSET + position,
                "source": "random",
            }
            for position, answer_id in enumerate(kept)
        )

    return pd.DataFrame(rows, columns=["question_id", "answer_id", "rank", "source"])


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

    answer_to_question, judged = build_exclusion_lookups(corpus, qrels)

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
