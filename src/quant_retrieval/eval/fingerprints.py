"""Stable identities for the exact tables used by an evaluation."""

import hashlib
import json

import pandas as pd


def _fingerprint(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()


def corpus_fingerprint(corpus: pd.DataFrame) -> str:
    return _fingerprint((int(row.answer_id), row.text) for row in corpus.itertuples())


def query_fingerprint(queries: pd.DataFrame) -> str:
    """Identify validated query IDs, text and row order independently of corpus files."""
    return _fingerprint((int(row.question_id), row.text) for row in queries.itertuples())


def dataset_fingerprints(
    corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame
) -> dict[str, str]:
    """Hash validated values in evaluation order, ignoring DataFrame index labels."""
    tables = {
        "corpus": ((int(row.answer_id), row.text) for row in corpus.itertuples()),
        "qrels": (
            (int(row.question_id), int(row.answer_id), int(row.grade))
            for row in qrels.itertuples()
        ),
    }
    return {"queries": query_fingerprint(queries),
            **{name: _fingerprint(rows) for name, rows in tables.items()}}
