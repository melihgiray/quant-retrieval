"""HTTP interface for the hybrid retrieval demo."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quant_retrieval.serve.search import SearchService
from quant_retrieval.serve.settings import ServeSettings

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


class HealthResponse(BaseModel):
    ready: bool
    documents: int
    pipeline: str


def service_from_environment() -> SearchService:
    """Build the service from paths that work locally and in the container."""
    settings = ServeSettings.from_environment()
    return SearchService.from_artifacts(
        checkpoint=settings.model_path,
        corpus_path=settings.corpus_path,
        manifest_path=settings.manifest_path,
        document_ids_path=settings.document_ids_path,
        embeddings_path=settings.embeddings_path,
        device=settings.device,
        depth=settings.depth,
        rrf_k=settings.rrf_k,
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

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        search_service = request.app.state.search_service
        return HealthResponse(
            ready=True,
            documents=search_service.document_count,
            pipeline=search_service.pipeline,
        )

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
