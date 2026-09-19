import json
from pathlib import Path

import pytest
from scripts.prepare_demo_snapshot import prepare_demo_snapshot

from quant_retrieval.serve.artifacts import (
    CHECKSUM_FILE,
    MODEL_FILES,
    REQUIRED_FILES,
    RETRIEVAL_FILES,
    verify_snapshot,
)


def source_files(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    artifacts = tmp_path / "artifacts"
    corpus = tmp_path / "corpus.parquet"
    checkpoint.mkdir()
    artifacts.mkdir()
    for name in MODEL_FILES:
        (checkpoint / name).write_bytes(f"model:{name}".encode())
    for name in RETRIEVAL_FILES:
        if name != "corpus.parquet":
            (artifacts / name).write_bytes(f"artifact:{name}".encode())
    corpus.write_bytes(b"corpus")
    return checkpoint, corpus, artifacts


def test_prepare_snapshot_copies_the_complete_contract(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)

    output = prepare_demo_snapshot(checkpoint, corpus, artifacts, tmp_path / "output")

    assert all((output / relative).is_file() for relative in REQUIRED_FILES)
    assert (output / "demo/corpus.parquet").read_bytes() == b"corpus"
    assert (output / CHECKSUM_FILE).is_file()
    assert not (output / f".{CHECKSUM_FILE}.tmp").exists()
    assert set(json.loads((output / CHECKSUM_FILE).read_text())) == set(REQUIRED_FILES) - {
        CHECKSUM_FILE
    }
    verify_snapshot(output)


def test_prepare_snapshot_checks_every_source_before_copying(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)
    (checkpoint / "model.safetensors").unlink()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        prepare_demo_snapshot(checkpoint, corpus, artifacts, output)

    assert not output.exists()


def test_prepare_snapshot_rejects_an_output_that_overwrites_sources(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)
    original = (checkpoint / "config.json").read_bytes()

    with pytest.raises(ValueError, match="overlaps source files"):
        prepare_demo_snapshot(checkpoint, corpus, artifacts, checkpoint)

    assert (checkpoint / "config.json").read_bytes() == original
    assert not (checkpoint / CHECKSUM_FILE).exists()


def test_prepare_snapshot_rejects_a_linked_output_directory(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "demo").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="linked paths"):
        prepare_demo_snapshot(checkpoint, corpus, artifacts, output)

    assert not list(outside.iterdir())


def test_prepare_snapshot_rejects_a_linked_output_root(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="linked paths"):
        prepare_demo_snapshot(checkpoint, corpus, artifacts, output)

    assert not list(outside.iterdir())


def test_verify_snapshot_rejects_linked_payload_files(tmp_path: Path):
    checkpoint, corpus, artifacts = source_files(tmp_path)
    output = prepare_demo_snapshot(checkpoint, corpus, artifacts, tmp_path / "output")
    target = output / "config.json"
    target.unlink()
    target.symlink_to(checkpoint / "config.json")

    with pytest.raises(RuntimeError, match="linked files"):
        verify_snapshot(output)
