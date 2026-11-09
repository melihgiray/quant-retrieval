import json
import sys

import pytest
from scripts import slice_run

from tests.test_query_slices import query_fixture
from tests.test_result_audit import diagnostic_record


def test_slice_command_saves_definition_and_protects_outputs(tmp_path, monkeypatch):
    source, output = tmp_path / "run.json", tmp_path / "groups.json"
    source.write_text(json.dumps(diagnostic_record()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slice",
            "--run",
            str(source),
            "--output",
            str(output),
            "--group-by",
            "first_primary_rank",
        ],
    )
    slice_run.main()
    report = json.loads(output.read_text())
    assert report["queries"] == 2
    assert report["grouping_definition"]["group_by"] == "first_primary_rank"
    previous = output.read_bytes()
    with pytest.raises(SystemExit):
        slice_run.main()
    assert output.read_bytes() == previous


def test_query_length_command_records_verified_source_and_word_units(tmp_path, monkeypatch):
    record, queries = query_fixture()
    source, table, output = (
        tmp_path / "run.json",
        tmp_path / "queries.parquet",
        tmp_path / "out.json",
    )
    source.write_text(json.dumps(record))
    queries.to_parquet(table)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slice",
            "--run",
            str(source),
            "--output",
            str(output),
            "--group-by",
            "query_length",
            "--queries",
            str(table),
            "--word-boundaries",
            "10",
            "25",
        ],
    )
    slice_run.main()
    report = json.loads(output.read_text())
    assert report["grouping_definition"]["word_boundaries"] == [10, 25]
    assert report["grouping_definition"]["length_unit"] == "whitespace_words"
    assert len(report["source"]["query_table"]["sha256"]) == 64
