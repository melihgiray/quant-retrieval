import hashlib

from quant_retrieval.serve.artifacts import PAYLOAD_FILES, write_checksums


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
