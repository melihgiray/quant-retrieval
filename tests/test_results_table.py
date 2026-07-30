import json
from pathlib import Path

import pytest
from scripts.make_results_table import load_evaluations, render_results


def result(name: str, retriever: str, score: float) -> dict:
    return {
        "run_name": name,
        "retriever": retriever,
        "split": "val",
        "config": {"parameters": {}},
        "metrics": {
            "ndcg_at_10": score,
            "mrr_at_10": score,
            "recall_at_10": score,
            "recall_at_100": score,
            "graded_ndcg_at_10": score,
        },
        "timing": {"index_seconds": 1.0, "search_ms_per_query_p50": 2.0},
    }


def test_loader_skips_dataset_stats_and_other_splits(tmp_path: Path):
    (tmp_path / "dataset_stats.json").write_text(json.dumps({"posts": {"questions": 1}}))
    (tmp_path / "val.json").write_text(json.dumps(result("bm25_val", "bm25", 0.4)))
    test_result = result("bm25_test", "bm25", 0.5)
    test_result["split"] = "test"
    (tmp_path / "test.json").write_text(json.dumps(test_result))

    loaded = load_evaluations(tmp_path, "val")
    assert [item["run_name"] for item in loaded] == ["bm25_val"]


def test_renderer_uses_measured_values_and_warns_that_test_is_untouched():
    markdown = render_results([result("bm25_val", "bm25", 0.41234)], "val")
    assert "| BM25 | 0.4123 |" in markdown
    assert "The test split has not been run." in markdown
    assert "26,152 answers" in markdown


def test_renderer_rejects_an_empty_result_set():
    with pytest.raises(ValueError, match="no val"):
        render_results([], "val")
