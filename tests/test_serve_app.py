import pandas as pd
from fastapi.testclient import TestClient

from quant_retrieval.retrieval.base import SearchResult
from quant_retrieval.serve.app import create_app
from quant_retrieval.serve.search import SearchService


class StubRetriever:
    def index(self, document_ids, texts):
        pass

    def search(self, query, k):
        return [SearchResult(document_id=20, score=0.75)][:k]


def client():
    corpus = pd.DataFrame(
        {"answer_id": [20], "question_id": [2], "text": ["volatility answer"]}
    )
    return TestClient(create_app(SearchService(StubRetriever(), corpus)))


def test_health_reports_a_loaded_service():
    with client() as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_search_returns_ranked_answers():
    with client() as test_client:
        response = test_client.get("/search", params={"q": "  volatility  ", "k": 1})

    assert response.status_code == 200
    assert response.json() == {
        "query": "volatility",
        "results": [
            {
                "answer_id": 20,
                "question_id": 2,
                "score": 0.75,
                "text": "volatility answer",
                "url": "https://quant.stackexchange.com/a/20",
            }
        ],
    }


def test_search_validates_query_and_result_count():
    with client() as test_client:
        blank = test_client.get("/search", params={"q": "   "})
        too_many = test_client.get("/search", params={"q": "delta", "k": 21})

    assert blank.status_code == 422
    assert too_many.status_code == 422
