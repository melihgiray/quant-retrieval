import pytest

from quant_retrieval.retrieval.checkpoint import (
    CHECKPOINT_FILES,
    checkpoint_hashes,
    validate_checkpoint_hashes,
    verify_checkpoint,
)


def make_checkpoint(path):
    path.mkdir()
    for name in CHECKPOINT_FILES:
        (path / name).write_bytes(name.encode())
    return path


def test_checkpoint_identity_survives_relocation_and_detects_tokenizer_changes(tmp_path):
    first = make_checkpoint(tmp_path / "first")
    second = make_checkpoint(tmp_path / "second")
    manifest = {"checkpoint_sha256": checkpoint_hashes(first)}
    assert verify_checkpoint(second, manifest)
    (second / "tokenizer_config.json").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint(second, manifest)


@pytest.mark.parametrize("change", ["missing", "empty", "link"])
def test_unusable_checkpoint_files_fail_closed(tmp_path, change):
    root = make_checkpoint(tmp_path / "model")
    weights = root / "model.safetensors"
    weights.unlink()
    if change == "empty":
        weights.touch()
    elif change == "link":
        weights.symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="nonempty regular"):
        checkpoint_hashes(root)


def test_optional_tokenizer_files_are_part_of_model_identity(tmp_path):
    root = make_checkpoint(tmp_path / "model")
    manifest = {"checkpoint_sha256": checkpoint_hashes(root)}
    (root / "added_tokens.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint(root, manifest)
    assert verify_checkpoint(root, {}) is False
    for invalid in (None, {}, {"../weights": "0" * 64}):
        with pytest.raises(ValueError, match="checksums"):
            verify_checkpoint(root, {"checkpoint_sha256": invalid})


def test_archived_identity_can_be_validated_without_loading_checkpoint_files():
    hashes = {name: "a" * 64 for name in CHECKPOINT_FILES}
    validate_checkpoint_hashes(hashes)
    for invalid in (None, {**hashes, "../weights": "b" * 64},
                    {**hashes, "model.safetensors": "not-a-digest"}):
        with pytest.raises(ValueError, match="checksums"):
            validate_checkpoint_hashes(invalid)
