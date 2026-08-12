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
import torch
from torch import Tensor

from quant_retrieval.data.pairs import GRADE_PRIMARY


@dataclass(frozen=True)
class TrainingPair:
    question_id: int
    query_text: str
    document_text: str
    negative_texts: tuple[str, ...] = ()


def load_training_pairs(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    split: str = "train",
    negatives: pd.DataFrame | None = None,
    negatives_per_query: int = 0,
    random_negatives_per_query: int = 0,
) -> list[TrainingPair]:
    """Join the three tables into (question, answer) text pairs for one split.

    With `negatives`, each pair also carries the wrong answers its question was
    given. `negatives_per_query` takes mined ones in rank order, hardest first,
    rather than sampling: the sampling variant would add diversity across epochs,
    but a fixed choice keeps a rerun of the same config on the same numbers,
    which matters more while runs are being compared.

    `random_negatives_per_query` adds uniformly drawn documents on top. Training
    on mined negatives alone teaches a model to separate answers that already
    look plausible and never teaches it to reject one that is off topic, which is
    what the reranker turned out to be missing.

    A question that cannot fill both quotas is dropped, because every example in
    a batch has to contribute the same number of candidates.
    """
    selected = queries.loc[queries["split"] == split]
    if selected.empty:
        raise ValueError(f"split {split!r} contains no queries")
    if negatives_per_query < 0 or random_negatives_per_query < 0:
        raise ValueError("negative counts cannot be negative")
    if (negatives_per_query or random_negatives_per_query) and negatives is None:
        raise ValueError("negatives were requested but no negatives were given")

    primary = qrels.loc[qrels["grade"] == GRADE_PRIMARY, ["question_id", "answer_id"]]
    documents = corpus.set_index("answer_id")["text"]

    joined = selected[["question_id", "text"]].merge(primary, on="question_id", how="inner")
    joined["document_text"] = joined["answer_id"].map(documents)
    joined = joined[joined["document_text"].notna()]

    wanted = negatives_per_query + random_negatives_per_query
    chosen = _negatives_by_question(
        negatives, documents, negatives_per_query, random_negatives_per_query
    )

    pairs = []
    for row in joined.itertuples(index=False):
        negative_texts = chosen.get(int(row.question_id), ()) if wanted else ()
        if wanted and len(negative_texts) < wanted:
            continue
        pairs.append(
            TrainingPair(
                question_id=int(row.question_id),
                query_text=row.text,
                document_text=row.document_text,
                negative_texts=negative_texts,
            )
        )
    return pairs


def _negatives_by_question(
    negatives: pd.DataFrame | None,
    documents: pd.Series,
    per_query: int,
    random_per_query: int = 0,
) -> dict[int, tuple[str, ...]]:
    """Pick each question's negatives: mined ones first, then random ones."""
    if negatives is None or per_query + random_per_query == 0:
        return {}
    ranked = negatives.sort_values(["question_id", "rank"])
    ranked = ranked[ranked["answer_id"].isin(documents.index)]
    is_random = ranked["source"] == "random"

    by_question: dict[int, tuple[str, ...]] = {}
    for question_id, group in ranked.groupby("question_id"):
        rows = group[~is_random.loc[group.index]].iloc[:per_query]
        if random_per_query:
            drawn = group[is_random.loc[group.index]].iloc[:random_per_query]
            rows = pd.concat([rows, drawn])
        by_question[int(question_id)] = tuple(documents.loc[rows["answer_id"]].tolist())
    return by_question


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
        batch = {
            "query_input_ids": queries["input_ids"],
            "query_attention_mask": queries["attention_mask"],
            "document_input_ids": documents["input_ids"],
            "document_attention_mask": documents["attention_mask"],
        }

        counts = {len(pair.negative_texts) for pair in pairs}
        if counts == {0}:
            return batch
        if len(counts) > 1:
            raise ValueError(f"every pair needs the same number of negatives, saw {sorted(counts)}")

        # Flattened to (batch * negatives, length). The encoder does not care,
        # and the training loop reshapes the embeddings back.
        flattened = [text for pair in pairs for text in pair.negative_texts]
        negatives = self._encode(flattened)
        batch["negative_input_ids"] = negatives["input_ids"]
        batch["negative_attention_mask"] = negatives["attention_mask"]
        return batch

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class CrossEncoderCollator:
    """Tokenize each pair as one sequence, grouped positive-first.

    Output is flat, (groups * candidates, length), with the group's positive at
    offset 0. The training loop reshapes the scores back to (groups, candidates).

    Truncation is `longest_first`, which trims whichever side is currently longer
    until the pair fits. Trimming only the answer was the first attempt and it
    fails outright here: plenty of questions on this site are longer than the
    whole budget on their own, so there is nothing left to take off the answer.
    Both sides lead with their most useful text, the question with its title and
    the answer with its opening, so trimming from the end of the longer one costs
    the least.
    """

    def __init__(self, tokenizer, max_length: int = 320) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, pairs: Sequence[TrainingPair]) -> dict[str, Tensor]:
        counts = {len(pair.negative_texts) for pair in pairs}
        if len(counts) > 1:
            raise ValueError(f"every group needs the same size, saw {sorted(counts)}")
        if counts == {0}:
            raise ValueError("the reranker needs at least one negative per question")

        queries: list[str] = []
        documents: list[str] = []
        for pair in pairs:
            for document in (pair.document_text, *pair.negative_texts):
                queries.append(pair.query_text)
                documents.append(document)

        encoded = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]
        batch["group_size"] = torch.tensor(counts.pop() + 1)
        return batch
