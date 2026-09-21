import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts import compare_runs
from scripts.compare_runs import load_per_query, validate_comparable_runs


@pytest.mark.parametrize("query_id", ["01", "-1", "0", "1.0", "one"])
def test_comparison_rejects_ids_that_cannot_be_paired_unambiguously(tmp_path, query_id):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {query_id: {"ndcg_at_10": 0.5}}}))
    with pytest.raises(SystemExit, match="invalid query ID"):
        load_per_query(path, "ndcg_at_10")


@pytest.mark.parametrize("value", [None, True, "0.5", -0.1, 1.1, float("nan")])
def test_comparison_rejects_invalid_metric_values(tmp_path, value):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {"1": {"ndcg_at_10": value}}}))
    with pytest.raises(SystemExit, match="invalid.*score"):
        load_per_query(path, "ndcg_at_10")


def test_comparison_loads_valid_query_scores(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"per_query": {"12": {"ndcg_at_10": 0.5}}}))
    assert load_per_query(path, "ndcg_at_10") == {12: 0.5}


@pytest.mark.parametrize("contents", [
    b'[]', b'{', b'\xff',
    b'{"per_query":{"1":{"ndcg_at_10":0.1},"1":{"ndcg_at_10":0.9}}}',
])
def test_comparison_rejects_malformed_or_ambiguous_json(tmp_path, contents):
    path = tmp_path / "run.json"
    path.write_bytes(contents)
    with pytest.raises(SystemExit, match="run.json"):
        load_per_query(path, "ndcg_at_10")


@pytest.mark.parametrize("counts", [None, [], 3])
def test_comparison_reports_malformed_counts(counts):
    with pytest.raises(SystemExit, match="counts object"):
        validate_comparable_runs({**run_record(), "counts": counts}, run_record())


def run_record():
    return {"split": "val", "counts": {"queries": 1, "corpus_documents": 10, "max_results": 100}}


@pytest.mark.parametrize("field", ["split", "queries", "corpus_documents", "max_results"])
def test_comparison_rejects_different_evaluation_conditions(field):
    baseline, candidate = run_record(), run_record()
    if field == "split":
        candidate[field] = "train"
    else:
        candidate["counts"][field] += 1
    with pytest.raises(SystemExit, match="comparison requires"):
        validate_comparable_runs(baseline, candidate)


def test_matching_evaluation_conditions_are_comparable():
    validate_comparable_runs(run_record(), run_record())


@pytest.mark.parametrize("field", ["corpus", "queries", "qrels"])
def test_comparison_rejects_changed_data_even_when_counts_match(field):
    hashes = {key: "a" * 64 for key in ("corpus", "queries", "qrels")}
    baseline = {**run_record(), "dataset_sha256": hashes}
    candidate = {**run_record(), "dataset_sha256": {**hashes, field: "b" * 64}}
    with pytest.raises(SystemExit, match="fingerprints differ"):
        validate_comparable_runs(baseline, candidate)
    validate_comparable_runs(baseline, baseline)


def test_comparison_cannot_mix_fingerprinted_and_legacy_records():
    baseline = {**run_record(), "dataset_sha256": {key: "a" * 64 for key in (
        "corpus", "queries", "qrels"
    )}}
    with pytest.raises(SystemExit, match="complete dataset fingerprints"):
        validate_comparable_runs(baseline, run_record())


def test_comparison_command_records_reproduction_inputs(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    for path, score in ((baseline, 0.2), (candidate, 0.4)):
        path.write_text(json.dumps({**run_record(), "per_query": {"1": {"ndcg_at_10": score}}}))
    output = tmp_path / "comparisons"
    monkeypatch.setattr(sys, "argv", [
        "compare_runs", "--baseline", str(baseline), "--candidate", str(candidate),
        "--out", str(output), "--seed", "8", "--iterations", "20", "--confidence", "0.8",
    ])
    compare_runs.main()
    record = json.loads(next(output.glob("*.json")).read_text())
    assert record["seed"] == 8
    assert record["confidence"] == 0.8
    assert record["iterations"] == 20
    assert record["split"] == "val"
    assert record["dataset_identity_verified"] is False
    assert record["source_sha256"]["baseline"] == hashlib.sha256(baseline.read_bytes()).hexdigest()

    report_path = next(output.glob("*.json"))
    original = report_path.read_bytes()

    def fail_replace(self, target):
        raise OSError("comparison write interrupted")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="comparison write interrupted"):
        compare_runs.main()
    assert report_path.read_bytes() == original
    assert list(output.iterdir()) == [report_path]
