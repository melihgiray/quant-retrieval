"""Stable identities for the exact tables used by an evaluation."""

import hashlib
import json

import pandas as pd


def dataset_fingerprints(
    corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame
) -> dict[str, str]:
    """Hash validated values in evaluation order, ignoring DataFrame index labels."""
    tables = {
        "corpus": ((int(row.answer_id), row.text) for row in corpus.itertuples()),
        "queries": ((int(row.question_id), row.text) for row in queries.itertuples()),
        "qrels": (
            (int(row.question_id), int(row.answer_id), int(row.grade))
            for row in qrels.itertuples()
        ),
    }
    result = {}
    for name, rows in tables.items():
        digest = hashlib.sha256()
        for row in rows:
            serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            digest.update(serialized.encode("utf-8"))
        result[name] = digest.hexdigest()
    return result
