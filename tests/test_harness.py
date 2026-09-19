from pathlib import Path

import pandas as pd
import pytest

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


def test_harness_rejects_duplicate_query_ids_in_a_split():
    corpus, queries, qrels = small_dataset()
    queries = pd.concat([queries, queries.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate question IDs"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [0, -1, 10.5])
def test_harness_rejects_invalid_query_ids(invalid):
    corpus, queries, qrels = small_dataset()
    queries["question_id"] = pd.Series([invalid, 20, 30])

    with pytest.raises(ValueError, match="query IDs must be positive integers"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [None, 12, "   "])
def test_harness_rejects_invalid_query_text(invalid):
    corpus, queries, qrels = small_dataset()
    queries["text"] = pd.Series([invalid, "volatility", "gamma"], dtype=object)

    with pytest.raises(ValueError, match="query texts must be nonempty strings"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


def test_harness_rejects_duplicate_corpus_ids():
    corpus, queries, qrels = small_dataset()
    corpus = pd.concat([corpus, corpus.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate answer IDs"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [0, -1, 1.5])
def test_harness_rejects_invalid_corpus_ids(invalid):
    corpus, queries, qrels = small_dataset()
    corpus["answer_id"] = pd.Series([invalid, 2, 3])

    with pytest.raises(ValueError, match="corpus answer IDs must be positive integers"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [None, 12, "   "])
def test_harness_rejects_invalid_corpus_text(invalid):
    corpus, queries, qrels = small_dataset()
    corpus["text"] = pd.Series([invalid, "gamma hedge", "volatility"], dtype=object)

    with pytest.raises(ValueError, match="corpus texts must be nonempty strings"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


def test_harness_rejects_duplicate_relevance_pairs():
    corpus, queries, qrels = small_dataset()
    qrels = pd.concat([qrels, qrels.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate question and answer pairs"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [0, -1, 10.5])
def test_harness_rejects_invalid_qrel_question_ids(invalid):
    corpus, queries, qrels = small_dataset()
    qrels["question_id"] = pd.Series([invalid, 20, 30])

    with pytest.raises(ValueError, match="qrel question IDs must be positive integers"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [0, -1, 1.5])
def test_harness_rejects_invalid_qrel_answer_ids(invalid):
    corpus, queries, qrels = small_dataset()
    qrels["answer_id"] = pd.Series([invalid, 3, 2])

    with pytest.raises(ValueError, match="qrel answer IDs must be positive integers"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize("invalid", [0, 3, 1.5])
def test_harness_rejects_invalid_qrel_grades(invalid):
    corpus, queries, qrels = small_dataset()
    qrels["grade"] = pd.Series([invalid, 2, 2])

    with pytest.raises(ValueError, match=r"qrel grades must be integers in \{1, 2\}"):
        evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)


@pytest.mark.parametrize(
    ("frame_name", "column"),
    [("corpus", "text"), ("queries", "split"), ("qrels", "grade")],
)
def test_harness_reports_missing_input_columns(frame_name, column):
    corpus, queries, qrels = small_dataset()
    frames = {"corpus": corpus, "queries": queries, "qrels": qrels}
    frames[frame_name] = frames[frame_name].drop(columns=column)

    with pytest.raises(ValueError, match=f"{frame_name} is missing columns.*{column}"):
        evaluate_retriever(
            KeywordRetriever(), frames["corpus"], frames["queries"], frames["qrels"]
        )


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (
            [SearchResult(document_id=1, score=1.0), SearchResult(document_id=1, score=0.5)],
            "duplicate document IDs",
        ),
        ([SearchResult(document_id=999, score=1.0)], "unknown document IDs"),
    ],
)
def test_harness_rejects_invalid_rankings(results, message):
    corpus, queries, qrels = small_dataset()
    retriever = KeywordRetriever()
    retriever.search = lambda query, k: results

    with pytest.raises(ValueError, match=message):
        evaluate_retriever(retriever, corpus, queries, qrels)


def test_harness_rejects_rankings_beyond_the_requested_cutoff():
    corpus, queries, qrels = small_dataset()
    retriever = KeywordRetriever()
    retriever.search = lambda query, k: [
        SearchResult(document_id=1, score=1.0) for _ in range(k + 1)
    ]

    with pytest.raises(ValueError, match="more results than requested"):
        evaluate_retriever(retriever, corpus, queries, qrels)


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
