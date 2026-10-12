import hashlib

import pytest

from quant_retrieval.serve.artifacts import PAYLOAD_FILES, verify_snapshot, write_checksums


def payload_fixture(root):
    for name in PAYLOAD_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return root


def test_checksum_writer_covers_optional_tokenizer_contents(tmp_path):
    root = payload_fixture(tmp_path)
    (root / "added_tokens.json").write_bytes(b'{"token": 3}')
    (root / "optimizer.pt").write_bytes(b"not an inference input")
    hashes = write_checksums(root)
    assert hashes["added_tokens.json"] == hashlib.sha256(b'{"token": 3}').hexdigest()
    assert "optimizer.pt" not in hashes
    assert set(PAYLOAD_FILES) < set(hashes)


def test_optional_tokenizers_are_verified_and_returned_in_the_upload_contract(tmp_path):
    root = payload_fixture(tmp_path)
    extra = root / "added_tokens.json"
    extra.write_text("{}")
    write_checksums(root)
    assert "added_tokens.json" in verify_snapshot(root)
    extra.write_text("changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_snapshot(root)


def test_undeclared_optional_files_cannot_bypass_snapshot_verification(tmp_path):
    root = payload_fixture(tmp_path)
    write_checksums(root)
    (root / "added_tokens.json").write_text("{}")
    with pytest.raises(RuntimeError, match="required payload"):
        verify_snapshot(root)
