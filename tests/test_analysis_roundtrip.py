import json
import subprocess
import sys

from quant_retrieval.eval.harness import evaluate_retriever
from quant_retrieval.eval.results import build_result_record
from tests.test_harness import KeywordRetriever, small_dataset


def test_real_evaluation_record_passes_audit_and_paired_slices_offline(tmp_path):
    corpus, queries, qrels = small_dataset()
    evaluation = evaluate_retriever(KeywordRetriever(), corpus, queries, qrels)
    record = build_result_record(
        "fixture", "keyword", "val", {}, evaluation, commit="abc1234", include_rankings=True
    )
    source, candidate, table = (
        tmp_path / name for name in ("baseline.json", "candidate.json", "queries.parquet")
    )
    source.write_text(json.dumps(record))
    candidate.write_text(json.dumps({**record, "run_name": "same-system"}))
    queries.to_parquet(table)
    audit, slices = tmp_path / "audit.json", tmp_path / "slices.json"
    for module, arguments in [
        ("scripts.audit_run", ["--run", str(source), "--output", str(audit)]),
        (
            "scripts.slice_run",
            [
                "--run",
                str(source),
                "--candidate",
                str(candidate),
                "--queries",
                str(table),
                "--group-by",
                "query_length",
                "--output",
                str(slices),
            ],
        ),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; module=sys.argv.pop(1); "
                "runpy.run_module(module,run_name='__main__'); "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
                module,
                *arguments,
            ],
            check=True,
        )
    report = json.loads(slices.read_text())
    assert report["dataset_identity_matched"] is True
    assert report["queries"] == evaluation["counts"]["queries"]
    assert sum(row["queries"] for row in report["groups"]) == report["queries"]
    assert report["ranking_overlap"]["mean_jaccard"] == 1
    assert all(all(value == 0 for value in row["delta"].values()) for row in report["groups"])
    assert json.loads(audit.read_text())["provenance"]["dataset_identity_recorded"] is True
