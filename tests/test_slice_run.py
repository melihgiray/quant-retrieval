import hashlib
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


def test_paired_slice_command_preserves_both_sources_and_baseline_grouping(tmp_path, monkeypatch):
    source, candidate, output = (tmp_path / name for name in ("base.json", "next.json", "out.json"))
    source.write_text(json.dumps(diagnostic_record()))
    candidate.write_text(json.dumps({**diagnostic_record(), "run_name": "next"}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slice",
            "--run",
            str(source),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--group-by",
            "first_primary_rank",
        ],
    )
    slice_run.main()
    report = json.loads(output.read_text())
    assert report["scope"] == "descriptive_paired_slices"
    assert report["grouping_definition"]["assignment_source"] == "baseline"
    assert len(report["source"]["candidate"]["sha256"]) == 64
    assert all(row["delta"]["ndcg_at_10"] == 0 for row in report["groups"])


def test_query_table_digest_describes_the_same_bytes_used_for_grouping(tmp_path, monkeypatch):
    record, queries = query_fixture()
    source, table, output = (
        tmp_path / name for name in ("run.json", "queries.parquet", "out.json")
    )
    source.write_text(json.dumps(record))
    queries.to_parquet(table)
    original = table.read_bytes()
    read_parquet = slice_run.pd.read_parquet

    def changed_after_read(contents):
        frame = read_parquet(contents)
        table.write_bytes(b"changed after capture")
        return frame

    monkeypatch.setattr(slice_run.pd, "read_parquet", changed_after_read)
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
        ],
    )
    slice_run.main()
    report = json.loads(output.read_text())
    assert report["source"]["query_table"]["sha256"] == hashlib.sha256(original).hexdigest()
