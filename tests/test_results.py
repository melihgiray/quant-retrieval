import json
from pathlib import Path

import pytest

from quant_retrieval.eval.results import build_result_record, write_result


def test_result_replacement_failure_preserves_previous_record(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    write_result({"score": 0.5}, target)

    def fail_replace(self, destination):
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        write_result({"score": 0.8}, target)
    assert json.loads(target.read_text()) == {"score": 0.5}
    assert list(tmp_path.iterdir()) == [target]


def test_result_writer_creates_parent_and_replaces_complete_record(tmp_path):
    target = tmp_path / "nested" / "result.json"
    write_result({"old": True}, target)
    write_result({"new": True}, target)
    assert json.loads(target.read_text()) == {"new": True}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_results_cannot_replace_valid_metrics(tmp_path, score):
    target = tmp_path / "result.json"
    write_result({"metrics": {"ndcg": 0.5}}, target)
    with pytest.raises(ValueError, match="JSON compliant"):
        write_result({"metrics": {"ndcg": score}}, target)
    assert json.loads(target.read_text())["metrics"]["ndcg"] == 0.5


def test_record_captures_configuration_before_the_next_experiment():
    config = {"model": {"depth": 10}}
    evaluation = {"metrics": {"ndcg": 0.5}, "timing": {}, "counts": {}}
    record = build_result_record("first", "tiny", "val", config, evaluation, commit="abc")
    config["model"]["depth"] = 100
    evaluation["metrics"]["ndcg"] = 0.9
    assert record["config"]["model"]["depth"] == 10
    assert record["metrics"]["ndcg"] == 0.5
