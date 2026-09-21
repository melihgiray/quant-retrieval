from quant_retrieval.eval.fingerprints import dataset_fingerprints
from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.eval.results import build_result_record
from tests.test_harness import KeywordRetriever, small_dataset


def test_fingerprints_capture_text_labels_and_corpus_order():
    corpus, queries, qrels = small_dataset()
    original = dataset_fingerprints(corpus, queries, qrels)
    assert dataset_fingerprints(corpus.set_axis([4, 5, 6]), queries, qrels) == original
    assert dataset_fingerprints(corpus.iloc[::-1], queries, qrels)["corpus"] != original["corpus"]
    corpus.loc[0, "text"] = "changed answer"
    queries.loc[0, "text"] = "changed question"
    qrels.loc[0, "grade"] = 1
    changed = dataset_fingerprints(corpus, queries, qrels)
    assert all(changed[key] != original[key] for key in original)


def test_saved_evaluation_fingerprints_only_selected_queries_and_judgements():
    corpus, queries, qrels = small_dataset()
    evaluation = evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)
    record = build_result_record("tiny", "keyword", "val", {}, evaluation, commit="abc")
    expected = dataset_fingerprints(corpus, queries.iloc[:2], qrels.iloc[:2])
    assert record["dataset_sha256"] == expected
    evaluation["dataset_sha256"]["corpus"] = "mutated"
    assert record["dataset_sha256"] == expected
