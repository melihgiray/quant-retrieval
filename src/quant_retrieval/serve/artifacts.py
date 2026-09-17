"""File contract shared by snapshot preparation and deployment."""

from __future__ import annotations

import hashlib
import json
import re
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
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def linked_paths(root: Path, relatives: tuple[str, ...]) -> list[str]:
    linked = []
    for relative in relatives:
        path = root
        for part in Path(relative).parts:
            path = path / part
            if path.is_symlink():
                linked.append(relative)
                break
    return linked


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> dict[str, str]:
    checksums = {relative: file_sha256(root / relative) for relative in PAYLOAD_FILES}
    destination = root / CHECKSUM_FILE
    temporary = root / f".{CHECKSUM_FILE}.tmp"
    try:
        temporary.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checksums


def verify_snapshot(root: Path) -> None:
    linked = linked_paths(root, REQUIRED_FILES)
    if linked:
        raise RuntimeError(f"asset repository contains linked files: {linked}")
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"asset repository is missing files: {missing}")
    try:
        expected = json.loads((root / CHECKSUM_FILE).read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("asset checksum file is not valid JSON") from error
    if not isinstance(expected, dict) or set(expected) != set(PAYLOAD_FILES):
        raise RuntimeError("asset checksum file does not match the required payload")
    invalid = [
        relative
        for relative, digest in expected.items()
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
    ]
    if invalid:
        raise RuntimeError(f"asset checksum file contains invalid SHA-256 values: {invalid}")
    mismatched = [
        relative
        for relative in PAYLOAD_FILES
        if expected.get(relative) != file_sha256(root / relative)
    ]
    if mismatched:
        raise RuntimeError(f"asset checksum mismatch: {mismatched}")
