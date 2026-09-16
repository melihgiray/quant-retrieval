from pathlib import Path

from scripts.publish_demo_snapshot import publish_demo_snapshot

from quant_retrieval.serve.artifacts import PAYLOAD_FILES, REQUIRED_FILES, write_checksums


class FakeApi:
    def __init__(self):
        self.created = None
        self.uploaded = None

    def create_repo(self, **kwargs):
        self.created = kwargs

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs
        return "https://huggingface.co/owner/model/commit/abc123"


def snapshot(tmp_path: Path) -> Path:
    for relative in PAYLOAD_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    write_checksums(tmp_path)
    return tmp_path


def test_publish_verifies_creates_and_uploads_one_model_repo(tmp_path: Path):
    api = FakeApi()

    url = publish_demo_snapshot("owner/model", snapshot(tmp_path), private=False, api=api)

    assert api.created == {
        "repo_id": "owner/model",
        "repo_type": "model",
        "private": False,
        "exist_ok": True,
    }
    assert api.uploaded["folder_path"] == str(tmp_path)
    assert api.uploaded["repo_id"] == "owner/model"
    assert api.uploaded["allow_patterns"] == list(REQUIRED_FILES)
    assert url.endswith("/commit/abc123")
