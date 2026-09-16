"""File contract shared by snapshot preparation and deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

RETRIEVAL_FILES = (
    "answer_ids.npy",
    "corpus.parquet",
    "embeddings_fp16.npy",
    "manifest.json",
)

PAYLOAD_FILES = MODEL_FILES + tuple(f"demo/{name}" for name in RETRIEVAL_FILES)
CHECKSUM_FILE = "checksums.json"
REQUIRED_FILES = PAYLOAD_FILES + (CHECKSUM_FILE,)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> dict[str, str]:
    checksums = {relative: file_sha256(root / relative) for relative in PAYLOAD_FILES}
    (root / CHECKSUM_FILE).write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return checksums


def verify_snapshot(root: Path) -> None:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"asset repository is missing files: {missing}")
    try:
        expected = json.loads((root / CHECKSUM_FILE).read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("asset checksum file is not valid JSON") from error
    if not isinstance(expected, dict) or set(expected) != set(PAYLOAD_FILES):
        raise RuntimeError("asset checksum file does not match the required payload")
    mismatched = [
        relative
        for relative in PAYLOAD_FILES
        if expected.get(relative) != file_sha256(root / relative)
    ]
    if mismatched:
        raise RuntimeError(f"asset checksum mismatch: {mismatched}")
