"""Content identity for the self-contained encoder checkpoints used by exports."""

import re
from pathlib import Path

from quant_retrieval.retrieval.index_artifacts import file_digest

CHECKPOINT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
TOKENIZER_EXTRAS = ("special_tokens_map.json", "added_tokens.json", "vocab.txt", "merges.txt",
                    "vocab.json", "tokenizer.model")


def checkpoint_hashes(directory: Path) -> dict[str, str]:
    """Hash inference files without loading weights or importing the model runtime."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("checkpoint must be a regular local directory")
    names = (*CHECKPOINT_FILES, *(name for name in TOKENIZER_EXTRAS
                                if (directory / name).exists() or (directory / name).is_symlink()))
    for name in names:
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint needs a nonempty regular file: {name}")
    return {name: file_digest(directory / name) for name in names}


def validate_checkpoint_hashes(expected: dict) -> None:
    """Check an archived identity without requiring local model files."""
    if (not isinstance(expected, dict) or not set(CHECKPOINT_FILES) <= set(expected)
            or not set(expected) <= set(CHECKPOINT_FILES + TOKENIZER_EXTRAS)
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in expected.values())):
        raise ValueError("checkpoint checksums must describe the encoder and tokenizer files")


def verify_checkpoint(directory: Path, manifest: dict) -> bool:
    """Legacy exports have no identity proof; present but invalid proofs must fail."""
    if "checkpoint_sha256" not in manifest:
        return False
    expected = manifest["checkpoint_sha256"]
    validate_checkpoint_hashes(expected)
    if checkpoint_hashes(directory) != expected:
        raise ValueError("checkpoint checksum mismatch: encoder or tokenizer changed")
    return True
