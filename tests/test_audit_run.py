import hashlib
import json
import subprocess
import sys

import pytest
from scripts import audit_run

from tests.test_result_audit import diagnostic_record


def test_audit_command_preserves_source_and_never_loads_a_model(tmp_path):
    source, output = tmp_path / "run.json", tmp_path / "audit.json"
    source.write_text(json.dumps(diagnostic_record()))
    original = source.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_module('scripts.audit_run', run_name='__main__'); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
            "--run",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["rankings"]["recorded"] is True
    assert report["relevance_recomputed"] is False
    assert source.read_bytes() == original


def test_audit_command_protects_existing_output_before_reading_input(tmp_path, monkeypatch):
    output = tmp_path / "keep.json"
    output.write_text("keep")
    monkeypatch.setattr(
        sys, "argv", ["audit", "--run", str(tmp_path / "missing"), "--output", str(output)]
    )
    with pytest.raises(SystemExit):
        audit_run.main()
    assert output.read_text() == "keep"
