from pathlib import Path

import pandas as pd

from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.eval.results import build_result_record, write_result
from quant_retrieval.retrieval.base import SearchResult


class KeywordRetriever:
    def index(self, document_ids: list[int], texts: list[str]) -> None:
        self.documents = dict(zip(document_ids, texts, strict=True))

    def search(self, query: str, k: int) -> list[SearchResult]:
        ranked = [
            SearchResult(document_id=document_id, score=float(query in text))
            for document_id, text in self.documents.items()
        ]
        return sorted(ranked, key=lambda result: (-result.score, result.document_id))[:k]


def small_dataset():
    corpus = pd.DataFrame(
        {"answer_id": [1, 2, 3], "text": ["delta hedge", "gamma hedge", "volatility"]}
    )
    queries = pd.DataFrame(
        {
            "question_id": [10, 20, 30],
            "text": ["delta", "volatility", "gamma"],
            "split": ["val", "val", "test"],
        }
    )
    qrels = pd.DataFrame(
        {
            "question_id": [10, 20, 30],
            "answer_id": [1, 3, 2],
            "grade": [2, 2, 2],
        }
    )
    return corpus, queries, qrels


def test_harness_indexes_full_corpus_and_only_scores_selected_split():
    corpus, queries, qrels = small_dataset()
    result = evaluate_retriever(KeywordRetriever(), corpus, queries, qrels, split="val")

    assert result["counts"] == {"corpus_documents": 3, "queries": 2, "max_results": 100}
    assert result["metrics"]["mrr_at_10"] == 1.0
    assert set(result["rankings"]) == {10, 20}
    assert result["timing"]["index_seconds"] >= 0


def test_harness_rejects_cutoff_below_reported_recall():
    corpus, queries, qrels = small_dataset()
    try:
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels, max_results=10)
    except ValueError as error:
        assert "Recall@100" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_harness_rejects_queries_without_judgements():
    corpus, queries, qrels = small_dataset()
    qrels = qrels[qrels["question_id"] != 20]
    try:
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)
    except ValueError as error:
        assert "20" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_result_record_and_writer_keep_provenance(tmp_path: Path):
    evaluation = {
        "metrics": {"mrr_at_10": 0.5},
        "timing": {"index_seconds": 1.2},
        "counts": {"queries": 2},
    }
    record = build_result_record(
        "tiny",
        "keyword",
        "val",
        {"seed": 17},
        evaluation,
        commit="abc123",
    )
    output = tmp_path / "result.json"
    write_result(record, output)

    assert record["commit"] == "abc123"
    assert record["config"] == {"seed": 17}
    assert '"mrr_at_10": 0.5' in output.read_text()
