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


def flat(markdown: str) -> str:
    """Collapse the wrapping, so an assertion is about words rather than layout."""
    return " ".join(markdown.split())


def named(run_name: str, score_value: float, order: int | None = None, **metrics) -> dict:
    item = result(run_name, "dense", score_value)
    item["metrics"].update(metrics)
    if order is not None:
        item["config"]["display_order"] = order
    return item


def test_rows_follow_display_order_not_filename():
    evaluations = [named("b_val", 0.5, order=20), named("a_val", 0.4, order=10)]
    table = render_results(evaluations, "val")
    assert table.index("A Val") < table.index("B Val")


def test_ablation_sections_are_skipped_when_their_runs_are_missing():
    markdown = render_results([result("bm25_val", "bm25", 0.4)], "val")
    assert "Hard negatives did not help" not in markdown
    assert "Batch size matters" not in markdown
    assert "Pooling is not a free choice" not in markdown


def test_hard_negative_section_reports_the_measured_deltas():
    evaluations = [
        named("minilm_tuned_epoch3_val", 0.50, recall_at_100=0.90),
        named("minilm_hardneg_epoch3_val", 0.51, recall_at_100=0.88),
    ]
    markdown = render_results(evaluations, "val")
    assert "+0.0100 nDCG@10" in markdown
    assert "-0.0200 Recall@100" in markdown


def test_batch_section_names_the_best_size_it_actually_saw():
    evaluations = [
        named("minilm_batch16_epoch3_val", 0.40),
        named("minilm_batch32_epoch3_val", 0.60),
        named("minilm_tuned_epoch3_val", 0.50),
    ]
    markdown = flat(render_results(evaluations, "val"))
    # The claim is about the climb from the smallest to the best size seen, not
    # a ranking of sizes whose gaps the data may not support.
    assert "climb from 16 to 32" in markdown
    assert "+0.2000" in markdown


def test_a_difference_that_fails_the_bootstrap_is_labelled_as_noise():
    evaluations = [
        named("bm25_val", 0.40),
        named("minilm_tuned_epoch3_val", 0.50),
        named("hybrid_val", 0.52),
    ]
    comparisons = {
        ("minilm_tuned_epoch3_val", "hybrid_val", "ndcg_at_10"): {
            "baseline": "minilm_tuned_epoch3_val", "candidate": "hybrid_val",
            "metric": "ndcg_at_10", "mean_difference": 0.02, "ci_low": -0.01,
            "ci_high": 0.05, "p_value": 0.15, "significant": False,
        }
    }
    markdown = flat(render_results(evaluations, "val", comparisons))
    assert "not distinguishable from noise" in markdown


def test_a_difference_that_survives_shows_its_interval():
    evaluations = [
        named("minilm_tuned_epoch3_val", 0.50),
        named("minilm_cls_epoch3_val", 0.42),
    ]
    comparisons = {
        ("minilm_tuned_epoch3_val", "minilm_cls_epoch3_val", "ndcg_at_10"): {
            "baseline": "minilm_tuned_epoch3_val", "candidate": "minilm_cls_epoch3_val",
            "metric": "ndcg_at_10", "mean_difference": -0.08, "ci_low": -0.10,
            "ci_high": -0.06, "p_value": 0.0001, "significant": True,
        }
    }
    markdown = flat(render_results(evaluations, "val", comparisons))
    assert "[-0.1000, -0.0600]" in markdown
    assert "not distinguishable" not in markdown


def test_pooling_section_only_claims_it_beat_frozen_when_it_did():
    beaten = [
        named("minilm_tuned_epoch3_val", 0.50),
        named("minilm_cls_epoch3_val", 0.40),
        named("minilm_frozen_val", 0.45),
    ]
    assert "lands below the untrained encoder" in render_results(beaten, "val")

    ahead = [
        named("minilm_tuned_epoch3_val", 0.50),
        named("minilm_cls_epoch3_val", 0.48),
        named("minilm_frozen_val", 0.45),
    ]
    assert "lands below the untrained encoder" not in render_results(ahead, "val")
