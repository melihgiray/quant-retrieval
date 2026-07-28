"""Build the three tables an IR dataset needs: corpus, queries, qrels.

The task is answer retrieval. A query is a question, the corpus is every answer
on the site, and the judgement says which answers belong to that question.

Grades:

    2   the accepted answer, or when the asker never accepted one, the clearly
        top voted answer (see ``PairConfig.min_top_score``)
    1   any other answer on the same question with a non negative score

Grade 1 exists because a user searching the site wants an answer to their
question, not specifically the one green tick. Keeping the siblings also stops
the metric from punishing a model that surfaces the second best answer first.
The evaluation harness can ignore grade 1 and score strict accepted-only
retrieval instead, so this choice does not get baked in here.

Only accepted answers are used as grade 2 for validation and test queries. The
top voted fallback is noisier, and noise in the labels you report on is worse
than a smaller evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_retrieval.data.clean import build_query_text, html_to_text

GRADE_PRIMARY = 2
GRADE_SIBLING = 1


@dataclass(frozen=True)
class PairConfig:
    # A top voted answer stands in for an accepted one only if it scored at
    # least this much. On this site a score of 2 already means several people
    # who read the question agreed.
    min_top_score: int = 2
    # Answers below this score are not treated as relevant at all.
    min_sibling_score: int = 0


def build_corpus(answers: pd.DataFrame) -> pd.DataFrame:
    """Every answer becomes a document. Nothing is filtered out.

    Short and low quality answers stay in. They make retrieval harder, which is
    the honest version of the task, and dropping them would be tuning the
    benchmark rather than the model.
    """
    corpus = pd.DataFrame(
        {
            "answer_id": answers["answer_id"],
            "question_id": answers["question_id"],
            "score": answers["score"],
            "text": answers["body_html"].map(html_to_text),
        }
    )
    return corpus[corpus["text"].str.len() > 0].reset_index(drop=True)


def build_queries(questions: pd.DataFrame) -> pd.DataFrame:
    """Every question becomes a candidate query. Filtered later to ones with labels."""
    return pd.DataFrame(
        {
            "question_id": questions["question_id"],
            "creation_date": questions["creation_date"],
            "title": questions["title"],
            "tags": questions["tags"],
            "text": [
                build_query_text(title, body)
                for title, body in zip(questions["title"], questions["body_html"], strict=True)
            ],
        }
    )


def build_qrels(
    questions: pd.DataFrame, answers: pd.DataFrame, config: PairConfig | None = None
) -> pd.DataFrame:
    """Judge each answer against its question."""
    config = config or PairConfig()

    has_accepted = questions["accepted_answer_id"].notna()
    accepted = (
        questions.loc[has_accepted, ["question_id", "accepted_answer_id"]]
        .astype({"accepted_answer_id": "int64"})
        .rename(columns={"accepted_answer_id": "answer_id"})
    )
    accepted["grade"] = GRADE_PRIMARY
    accepted["label_source"] = "accepted"

    # For questions with no accepted answer, promote the single top voted answer
    # when it clears the bar and is not tied with the runner up.
    ranked = answers.sort_values(
        ["question_id", "score", "answer_id"], ascending=[True, False, True]
    )
    unaccepted = ranked[~ranked["question_id"].isin(accepted["question_id"])]
    best = unaccepted.groupby("question_id", as_index=False).head(1)
    second = unaccepted.groupby("question_id", as_index=False).nth(1)
    runner_up = second.set_index("question_id")["score"]

    best = best[best["score"] >= config.min_top_score].copy()
    best["runner_up_score"] = best["question_id"].map(runner_up).fillna(-999)
    best = best[best["score"] > best["runner_up_score"]]

    promoted = best[["question_id", "answer_id"]].copy()
    promoted["grade"] = GRADE_PRIMARY
    promoted["label_source"] = "top_voted"

    primary = pd.concat([accepted, promoted], ignore_index=True)

    # Everything else on a judged question, if it was not voted down.
    primary_ids = set(primary["answer_id"])
    siblings = answers[
        answers["question_id"].isin(set(primary["question_id"]))
        & ~answers["answer_id"].isin(primary_ids)
        & (answers["score"] >= config.min_sibling_score)
    ][["question_id", "answer_id"]].copy()
    siblings["grade"] = GRADE_SIBLING
    siblings["label_source"] = "sibling"

    qrels = pd.concat([primary, siblings], ignore_index=True)
    # An answer that no longer exists in the corpus cannot be a judgement.
    qrels = qrels[qrels["answer_id"].isin(set(answers["answer_id"]))]
    qrels = qrels.sort_values(["question_id", "grade"], ascending=[True, False])
    return qrels.reset_index(drop=True)
