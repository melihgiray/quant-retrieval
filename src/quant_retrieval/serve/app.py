"""HTTP interface for the hybrid retrieval demo."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quant_retrieval.serve.search import SearchService

STATIC_DIR = Path(__file__).parent / "static"


class SearchHitResponse(BaseModel):
    answer_id: int
    question_id: int
    score: float
    text: str
    url: str


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    results: list[SearchHitResponse]


def service_from_environment() -> SearchService:
    """Build the service from paths that work locally and in the container."""
    return SearchService.from_artifacts(
        checkpoint=Path(os.getenv("MODEL_PATH", "checkpoints/minilm_tuned/epoch-3")),
        corpus_path=Path(os.getenv("CORPUS_PATH", "data/processed/corpus.parquet")),
        document_ids_path=Path(os.getenv("DOCUMENT_IDS_PATH", "artifacts/answer_ids.npy")),
        embeddings_path=Path(
            os.getenv("EMBEDDINGS_PATH", "artifacts/embeddings_fp16.npy")
        ),
        device=os.getenv("DEVICE", "auto"),
    )


def create_app(service: SearchService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.search_service = service or service_from_environment()
        yield

    app = FastAPI(title="Quant Retrieval", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, bool]:
        return {"ready": hasattr(request.app.state, "search_service")}

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=1000),
        k: int = Query(default=10, ge=1, le=20),
    ) -> SearchResponse:
        try:
            started = time.perf_counter()
            hits = request.app.state.search_service.search(q, k=k)
            elapsed_ms = (time.perf_counter() - started) * 1000
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return SearchResponse(
            query=q.strip(),
            elapsed_ms=round(elapsed_ms, 2),
            results=[SearchHitResponse(**hit.to_dict()) for hit in hits],
        )

    return app


app = create_app()
