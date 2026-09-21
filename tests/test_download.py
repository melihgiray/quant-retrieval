import io

import pytest

from quant_retrieval.data import download


class Response(io.BytesIO):
    def __init__(self, body=b"new archive", headers=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))} if headers is None else headers


def test_complete_download_replaces_archive(tmp_path, monkeypatch):
    target = tmp_path / "dump.7z"
    target.write_bytes(b"old")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Response())
    info = download.download_dump(target, force=True)
    assert target.read_bytes() == b"new archive"
    assert info.bytes_downloaded == len(b"new archive")
    assert list(tmp_path.iterdir()) == [target]


def test_truncated_download_preserves_previous_archive(tmp_path, monkeypatch):
    target = tmp_path / "dump.7z"
    target.write_bytes(b"old")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Response(
        b"short", {"Content-Length": "100"}
    ))
    with pytest.raises(OSError, match="truncated"):
        download.download_dump(target, force=True)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_connection_failure_leaves_no_partial_archive(tmp_path, monkeypatch):
    class BrokenResponse(Response):
        def read(self, size=-1):
            raise OSError("connection lost")

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: BrokenResponse())
    with pytest.raises(OSError, match="connection lost"):
        download.download_dump(tmp_path / "dump.7z")
    assert list(tmp_path.iterdir()) == []


def test_download_uses_a_socket_timeout(tmp_path, monkeypatch):
    def open_response(request, *, timeout):
        assert timeout == 4.5
        return Response()

    monkeypatch.setattr(download.urllib.request, "urlopen", open_response)
    download.download_dump(tmp_path / "dump.7z", timeout=4.5)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_download_rejects_invalid_timeouts_before_creating_directories(tmp_path, timeout):
    target = tmp_path / "new" / "dump.7z"
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        download.download_dump(target, timeout=timeout)
    assert not target.parent.exists()


def test_missing_content_length_does_not_reuse_an_empty_archive(tmp_path, monkeypatch):
    target = tmp_path / "dump.7z"
    target.touch()
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Response(headers={}))
    download.download_dump(target)
    assert target.read_bytes() == b"new archive"


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "0"}, {"Content-Length": "bad"}])
def test_empty_or_invalid_response_preserves_archive(tmp_path, monkeypatch, headers):
    target = tmp_path / "dump.7z"
    target.write_bytes(b"old")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Response(b"", headers))
    with pytest.raises(OSError):
        download.download_dump(target, force=True)
    assert target.read_bytes() == b"old"
