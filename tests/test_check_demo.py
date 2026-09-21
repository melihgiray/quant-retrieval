import io
import json

import pytest
from scripts import check_demo

from tests.test_serve_app import client


def health():
    return {"ready": True, "pipeline": "bm25_dense_rrf", "documents": 10, "artifact_commit": "abc"}


def search():
    return {"query": "delta", "elapsed_ms": 1.5, "results": [{
        "answer_id": 10, "question_id": 1, "score": 0.5, "text": "answer",
        "url": "https://quant.stackexchange.com/a/10",
    }]}


def responses(monkeypatch, payloads):
    pending = iter(payloads)
    calls = []

    def open_response(request, *, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(json.dumps(next(pending)).encode())

    monkeypatch.setattr(check_demo.urllib.request, "urlopen", open_response)
    return calls


def test_probe_checks_readiness_then_encoded_search(monkeypatch):
    calls = responses(monkeypatch, [health(), search()])
    report = check_demo.check_demo("https://demo.example/", query=" delta ", expected_commit="abc")
    assert report["search"]["results"][0]["answer_id"] == 10
    assert calls == [
        ("https://demo.example/health", 30.0),
        ("https://demo.example/search?q=delta&k=3", 30.0),
    ]


def test_probe_stops_before_search_when_artifacts_are_wrong(monkeypatch):
    calls = responses(monkeypatch, [health()])
    with pytest.raises(ValueError, match="artifact commit"):
        check_demo.check_demo("https://demo.example", expected_commit="different")
    assert len(calls) == 1


@pytest.mark.parametrize("change", [
    {"results": []}, {"query": "other"}, {"elapsed_ms": float("nan")},
    {"results": search()["results"] * 2},
    {"results": [{**search()["results"][0], "score": float("inf")}]},
])
def test_probe_rejects_broken_search_contracts(monkeypatch, change):
    responses(monkeypatch, [health(), {**search(), **change}])
    with pytest.raises(ValueError):
        check_demo.check_demo("https://demo.example", query="delta")


@pytest.mark.parametrize("url", ["file:///tmp/data", "https://user:pass@example.com", "https://a/?q=x"])
def test_probe_rejects_unsupported_base_urls(url):
    with pytest.raises(ValueError, match="base URL"):
        check_demo.check_demo(url)


def test_probe_accepts_the_actual_app_contract(monkeypatch):
    with client() as app_client:
        def open_response(request, *, timeout):
            response = app_client.get(request.full_url)
            response.raise_for_status()
            return io.BytesIO(response.content)

        monkeypatch.setattr(check_demo.urllib.request, "urlopen", open_response)
        report = check_demo.check_demo("http://testserver", query="volatility")
    assert report["health"]["documents"] == 1
    assert report["search"]["results"][0]["answer_id"] == 20
