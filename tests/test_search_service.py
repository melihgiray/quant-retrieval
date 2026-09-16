import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from quant_retrieval.retrieval.base import SearchResult
from quant_retrieval.serve.search import ArtifactManifest, SearchService, make_snippet


class StubRetriever:
    def index(self, document_ids, texts):
        raise AssertionError("the service must receive an indexed retriever")

    def search(self, query, k):
        return [SearchResult(document_id=20, score=0.75)][:k]


def corpus():
    return pd.DataFrame(
        {
            "answer_id": [10, 20],
            "question_id": [1, 2],
            "text": ["delta answer", "volatility answer"],
        }
    )


def test_search_service_attaches_corpus_fields_and_source_link():
    hits = SearchService(StubRetriever(), corpus()).search("volatility", k=1)

    assert [hit.to_dict() for hit in hits] == [
        {
            "answer_id": 20,
            "question_id": 2,
            "score": 0.75,
            "text": "volatility answer",
            "url": "https://quant.stackexchange.com/a/20",
        }
    ]


@pytest.mark.parametrize(("query", "k"), [("", 10), ("   ", 10), ("delta", 0), ("delta", 21)])
def test_search_service_rejects_invalid_requests(query, k):
    with pytest.raises(ValueError):
        SearchService(StubRetriever(), corpus()).search(query, k=k)


def test_search_service_rejects_unknown_ranked_answers():
    service = SearchService(StubRetriever(), corpus())
    service.retriever.search = lambda query, k: [SearchResult(document_id=99, score=1.0)]

    with pytest.raises(RuntimeError, match="unknown answer 99"):
        service.search("delta")


def test_snippet_collapses_whitespace_and_stops_at_a_word():
    assert make_snippet("  delta\n hedge   explanation ", max_chars=16) == "delta hedge…"


def test_snippet_keeps_short_text_unchanged():
    assert make_snippet("short answer", max_chars=20) == "short answer"


def test_artifact_manifest_loads_index_shape(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": 100, "dimensions": 384, "max_length": 256}))

    assert ArtifactManifest.load(path) == ArtifactManifest(100, 384, 256)


def test_artifact_manifest_rejects_missing_shape(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": 100}))

    with pytest.raises(ValueError, match="dimensions"):
        ArtifactManifest.load(path)


@pytest.mark.parametrize(
    "contents",
    ["not JSON", "[]", '{"documents": 1.5, "dimensions": 384, "max_length": 256}',
     '{"documents": true, "dimensions": 384, "max_length": 256}'],
)
def test_artifact_manifest_rejects_invalid_records(tmp_path: Path, contents: str):
    path = tmp_path / "manifest.json"
    path.write_text(contents)

    with pytest.raises(ValueError, match="manifest"):
        ArtifactManifest.load(path)


def test_search_service_serializes_shared_retriever_calls():
    class ConcurrentProbe(StubRetriever):
        def __init__(self):
            self.state_lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def search(self, query, k):
            with self.state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.state_lock:
                self.active -= 1
            return [SearchResult(document_id=20, score=0.75)]

    retriever = ConcurrentProbe()
    service = SearchService(retriever, corpus())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(service.search, f"query {index}") for index in range(2)]
        assert all(future.result() for future in futures)

    assert retriever.max_active == 1
