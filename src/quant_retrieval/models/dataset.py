"""Training pairs and the collator that turns them into tensors.

A pair is one question and the answer judged primary for it. Only grade 2
judgements are used, and only from the training split. The grade 1 siblings
exist for evaluation, where partial credit is the point; as training targets
they would teach the model that a question maps to several answers at once,
which is not what the in-batch loss is set up to learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from torch import Tensor

from quant_retrieval.data.pairs import GRADE_PRIMARY


@dataclass(frozen=True)
class TrainingPair:
    question_id: int
    query_text: str
    document_text: str


def load_training_pairs(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    split: str = "train",
) -> list[TrainingPair]:
    """Join the three tables into (question, answer) text pairs for one split."""
    selected = queries.loc[queries["split"] == split]
    if selected.empty:
        raise ValueError(f"split {split!r} contains no queries")

    primary = qrels.loc[qrels["grade"] == GRADE_PRIMARY, ["question_id", "answer_id"]]
    documents = corpus.set_index("answer_id")["text"]

    joined = selected[["question_id", "text"]].merge(primary, on="question_id", how="inner")
    joined["document_text"] = joined["answer_id"].map(documents)
    joined = joined[joined["document_text"].notna()]

    return [
        TrainingPair(
            question_id=int(row.question_id),
            query_text=row.text,
            document_text=row.document_text,
        )
        for row in joined.itertuples(index=False)
    ]


class PairCollator:
    """Tokenize a batch of pairs into query and document tensors."""

    def __init__(self, tokenizer, max_length: int = 256) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, pairs: Sequence[TrainingPair]) -> dict[str, Tensor]:
        queries = self._encode([pair.query_text for pair in pairs])
        documents = self._encode([pair.document_text for pair in pairs])
        return {
            "query_input_ids": queries["input_ids"],
            "query_attention_mask": queries["attention_mask"],
            "document_input_ids": documents["input_ids"],
            "document_attention_mask": documents["attention_mask"],
        }

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
