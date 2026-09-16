import subprocess
import sys
from pathlib import Path

import pytest
from scripts import download_demo_assets as assets
from scripts import start_demo
from scripts.start_demo import configure_asset_environment

from quant_retrieval.serve.artifacts import (
    CHECKSUM_FILE,
    MODEL_FILES,
    PAYLOAD_FILES,
    REQUIRED_FILES,
    RETRIEVAL_FILES,
    write_checksums,
)


def fake_snapshot(
    output: Path,
    missing: str | None = None,
    corrupt: str | None = None,
    checksum_contents: str | None = None,
):
    def download(**kwargs):
        assert kwargs["repo_id"] == "owner/model"
        assert kwargs["revision"] == "abc123"
        for relative in PAYLOAD_FILES:
            if relative == missing:
                continue
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        if missing != CHECKSUM_FILE:
            if missing is None:
                write_checksums(output)
            else:
                (output / CHECKSUM_FILE).write_text("{}")
        if corrupt:
            (output / corrupt).write_bytes(b"changed after checksums")
        if checksum_contents is not None:
            (output / CHECKSUM_FILE).write_text(checksum_contents)
        return str(output)

    return download


def test_download_checks_and_returns_the_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot(tmp_path))

    downloaded = assets.download_demo_assets("owner/model", tmp_path, "abc123")

    assert downloaded == tmp_path
    assert all((downloaded / relative).is_file() for relative in REQUIRED_FILES)


def test_download_passes_a_private_repository_token(tmp_path: Path, monkeypatch):
    def download(**kwargs):
        assert kwargs["token"] == "fixture-token"
        return fake_snapshot(tmp_path)(**kwargs)

    monkeypatch.setattr(assets, "snapshot_download", download)

    assert assets.download_demo_assets(
        "owner/model", tmp_path, "abc123", token="fixture-token"
    ) == tmp_path


def test_download_rejects_an_incomplete_snapshot(tmp_path: Path, monkeypatch):
    missing = "demo/manifest.json"
    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot(tmp_path, missing))

    with pytest.raises(RuntimeError, match="manifest.json"):
        assets.download_demo_assets("owner/model", tmp_path, "abc123")


def test_download_rejects_a_checksum_mismatch(tmp_path: Path, monkeypatch):
    corrupted = "demo/embeddings_fp16.npy"
    monkeypatch.setattr(assets, "snapshot_download", fake_snapshot(tmp_path, corrupt=corrupted))

    with pytest.raises(RuntimeError, match="embeddings_fp16.npy"):
        assets.download_demo_assets("owner/model", tmp_path, "abc123")


@pytest.mark.parametrize("contents", ["not json", "[]", '{"config.json": "abc"}'])
def test_download_rejects_a_malformed_checksum_record(tmp_path: Path, monkeypatch, contents):
    monkeypatch.setattr(
        assets, "snapshot_download", fake_snapshot(tmp_path, checksum_contents=contents)
    )

    with pytest.raises(RuntimeError, match="checksum"):
        assets.download_demo_assets("owner/model", tmp_path, "abc123")


def test_remote_snapshot_paths_configure_the_server(tmp_path: Path):
    environ = {"DEVICE": "cpu"}

    configure_asset_environment(tmp_path, environ)

    assert environ["MODEL_PATH"] == str(tmp_path)
    assert environ["CORPUS_PATH"] == str(tmp_path / "demo/corpus.parquet")
    assert environ["EMBEDDINGS_PATH"] == str(tmp_path / "demo/embeddings_fp16.npy")
    assert environ["DEVICE"] == "cpu"


def test_launcher_forwards_hub_token_without_printing_it(tmp_path: Path, monkeypatch):
    calls = []

    def download(repo_id, output, revision, *, token):
        calls.append((repo_id, output, revision, token))
        return tmp_path

    monkeypatch.setattr(start_demo, "download_demo_assets", download)
    monkeypatch.setattr(start_demo, "configure_asset_environment", lambda root: None)
    monkeypatch.setattr(start_demo.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["start_demo", "--asset-repo", "owner/model"])
    monkeypatch.setenv("HF_TOKEN", "fixture-token")

    start_demo.main()

    assert calls == [("owner/model", Path("demo_assets"), "main", "fixture-token")]


def test_documented_module_entry_point_starts_from_repo_root():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.start_demo", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--asset-repo" in completed.stdout


def test_snapshot_contract_places_only_retrieval_files_under_demo():
    assert set(REQUIRED_FILES) == (
        set(MODEL_FILES)
        | {f"demo/{name}" for name in RETRIEVAL_FILES}
        | {CHECKSUM_FILE}
    )
