import sys

import pytest
from scripts import inspect_snapshot


def test_inspection_preserves_existing_output_before_loading_assets(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["inspect", "--snapshot", str(tmp_path / "missing"),
                                     "--output", str(output)])
    with pytest.raises(FileExistsError, match="already exists"):
        inspect_snapshot.main()
    assert output.read_text() == "keep"


def test_inspection_report_cannot_modify_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inspect", "--snapshot", str(tmp_path),
                                     "--output", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="outside the snapshot"):
        inspect_snapshot.main()
