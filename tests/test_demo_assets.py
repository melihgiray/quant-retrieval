from pathlib import Path

import pytest
from scripts import download_demo_assets as assets


def fake_snapshot(output: Path, missing: str | None = None):
    def download(**kwargs):
        assert kwargs["repo_id"] == "owner/model"
        assert kwargs["revision"] == "abc123"
        for relative in assets.REQUIRED_FILES:
            if relative == missing:
                continue
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return str(output)

    return download


def test_download_checks_and_returns_the_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot(tmp_path))

    downloaded = assets.download_demo_assets("owner/model", tmp_path, "abc123")

    assert downloaded == tmp_path
    assert all((downloaded / relative).is_file() for relative in assets.REQUIRED_FILES)


def test_download_rejects_an_incomplete_snapshot(tmp_path: Path, monkeypatch):
    missing = "demo/manifest.json"
    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot(tmp_path, missing))

    with pytest.raises(RuntimeError, match="manifest.json"):
        assets.download_demo_assets("owner/model", tmp_path, "abc123")
