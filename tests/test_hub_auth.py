from pathlib import Path

import pytest
from scripts.hub_auth import read_token_file


def test_token_file_is_trimmed(tmp_path: Path):
    path = tmp_path / "token"
    path.write_text("  fixture-token\n")

    assert read_token_file(path) == "fixture-token"


def test_missing_token_file_has_a_clear_error(tmp_path: Path):
    path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=f"token file does not exist: {path}"):
        read_token_file(path)


def test_empty_token_file_has_a_clear_error(tmp_path: Path):
    path = tmp_path / "token"
    path.write_text(" \n")

    with pytest.raises(RuntimeError, match=f"token file is empty: {path}"):
        read_token_file(path)
