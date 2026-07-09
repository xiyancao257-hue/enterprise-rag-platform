from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from enterprise_rag.config import AppConfig, load_config
from enterprise_rag.models import RagAnswer, SearchHit
from enterprise_rag.observability.tracing import QueryTrace, TraceHit
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.factory import create_vector_index

DEFAULT_INDEX = Path("data/processed/chunks.json")
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
logger = logging.getLogger("enterprise_rag.api")


class MetricsCollector:
    def __init__(self) -> None:
        self.http_requests_total = 0
        self.query_requests_total = 0
        self.query_failures_total = 0
        self.query_latency_ms_sum = 0.0
        self.query_latency_ms_count = 0
        self.query_citations_total = 0

    def record_http_request(self) -> None:
        self.http_requests_total += 1

    def record_query_success(self, latency_ms: float, citation_count: int) -> None:
        self.query_requests_total += 1
        self.query_latency_ms_sum += latency_ms
        self.query_latency_ms_count += 1
        self.query_citations_total += citation_count

    def record_query_failure(self) -> None:
        self.query_requests_total += 1
        self.query_failures_total += 1

    def render_prometheus(self) -> str:
        metrics = {
            "enterprise_rag_http_requests_total": self.http_requests_total,
            "enterprise_rag_query_requests_total": self.query_requests_total,
            "enterprise_rag_query_failures_total": self.query_failures_total,
            "enterprise_rag_query_latency_ms_sum": round(self.query_latency_ms_sum, 4),
            "enterprise_rag_query_latency_ms_count": self.query_latency_ms_count,
            "enterprise_rag_query_citations_total": self.query_citations_total,
        }
        return "\n".join(f"{name} {value}" for name, value in metrics.items()) + "\n"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    user_groups: list[str] = Field(default_factory=list)
    include_trace: bool = False


class QueryPlanResponse(BaseModel):
    original_query: str
    normalized_query: str
    rewritten_queries: list[str]
    ambiguity_notes: list[str]
    corrections: dict[str, str]
    metadata_filters: dict[str, str]


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    retriever: str
    rank: int
    heading_path: list[str]
    metadata: dict[str, str]


class TraceHitResponse(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    retriever: str
    rank: int
    heading_path: list[str]
    source_path: str | None = None


class QueryTraceResponse(BaseModel):
    original_query: str
    normalized_query: str
    rewritten_queries: list[str]
    metadata_filters: dict[str, str]
    retrieved: list[TraceHitResponse]
    reranked: list[TraceHitResponse]
    final_context: list[TraceHitResponse]


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    query_plan: QueryPlanResponse
    citations: list[CitationResponse]
    trace: QueryTraceResponse | None = None


class HealthResponse(BaseModel):
    request_id: str
    status: str
    chunk_count: int
    vector_index_provider: str


def create_app(
    index_path: Path = DEFAULT_INDEX,
    config_path: Path | None = None,
    config: AppConfig | None = None,
) -> FastAPI:
    config = config or load_config(config_path)
    app = FastAPI(title="Enterprise RAG API", version="0.1.0")
    app.state.index_path = index_path
    app.state.config = config
    app.state.metrics = MetricsCollector()

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or f"req_{uuid4().hex}"
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started_at) * 1000
        app.state.metrics.record_http_request()
        response.headers[REQUEST_ID_HEADER] = request_id
        _log_event(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )
        return response

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        _authorize_request(request, app.state.config)
        return app.state.metrics.render_prometheus()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        chunks = JsonChunkStore(app.state.index_path).load()
        return HealthResponse(
            request_id=request.state.request_id,
            status="ok",
            chunk_count=len(chunks),
            vector_index_provider=app.state.config.vector_index.provider,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        _authorize_request(request, app.state.config)
        started_at = time.perf_counter()
        chunks = JsonChunkStore(app.state.index_path).load()
        if not chunks:
            app.state.metrics.record_query_failure()
            _log_event(
                "query_failed",
                request_id=request.state.request_id,
                reason="empty_index",
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            raise HTTPException(status_code=404, detail="No chunks found. Run ingestion before querying.")

        config: AppConfig = app.state.config
        top_k = payload.top_k if payload.top_k is not None else config.retrieval.top_k
        pipeline = RagPipeline(
            chunks,
            enable_graph=config.retrieval.enable_graph,
            graph_max_hops=config.retrieval.graph_max_hops,
            vector_index=create_vector_index(config.vector_index),
        )
        answer, trace = pipeline.answer_for_user_with_trace(
            payload.query,
            top_k=top_k,
            user_groups=set(payload.user_groups) or set(config.security.default_user_groups),
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        app.state.metrics.record_query_success(latency_ms=latency_ms, citation_count=len(answer.citations))
        _log_event(
            "query_completed",
            request_id=request.state.request_id,
            latency_ms=round(latency_ms, 2),
            top_k=top_k,
            citation_count=len(answer.citations),
            include_trace=payload.include_trace,
            vector_index_provider=config.vector_index.provider,
        )
        return _query_response(request.state.request_id, answer, trace if payload.include_trace else None)

    return app


app = create_app()


def _query_response(request_id: str, answer: RagAnswer, trace: QueryTrace | None) -> QueryResponse:
    return QueryResponse(
        request_id=request_id,
        answer=answer.answer,
        query_plan=QueryPlanResponse(
            original_query=answer.query_plan.original_query,
            normalized_query=answer.query_plan.normalized_query,
            rewritten_queries=list(answer.query_plan.rewritten_queries),
            ambiguity_notes=list(answer.query_plan.ambiguity_notes),
            corrections=answer.query_plan.corrections,
            metadata_filters=answer.query_plan.metadata_filters,
        ),
        citations=[_citation_response(hit) for hit in answer.citations],
        trace=_trace_response(trace) if trace is not None else None,
    )


def _citation_response(hit: SearchHit) -> CitationResponse:
    return CitationResponse(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        text=hit.chunk.text,
        score=hit.score,
        retriever=hit.retriever,
        rank=hit.rank,
        heading_path=list(hit.chunk.heading_path),
        metadata=hit.chunk.metadata,
    )


def _trace_response(trace: QueryTrace) -> QueryTraceResponse:
    return QueryTraceResponse(
        original_query=trace.original_query,
        normalized_query=trace.normalized_query,
        rewritten_queries=list(trace.rewritten_queries),
        metadata_filters=trace.metadata_filters,
        retrieved=[_trace_hit_response(hit) for hit in trace.retrieved],
        reranked=[_trace_hit_response(hit) for hit in trace.reranked],
        final_context=[_trace_hit_response(hit) for hit in trace.final_context],
    )


def _trace_hit_response(hit: TraceHit) -> TraceHitResponse:
    return TraceHitResponse(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        score=hit.score,
        retriever=hit.retriever,
        rank=hit.rank,
        heading_path=list(hit.heading_path),
        source_path=hit.source_path,
    )


def _log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def _authorize_request(request: Request, config: AppConfig) -> None:
    if not config.api_security.require_api_key:
        return

    provided_key = _extract_api_key(request)
    if provided_key is None or not _is_valid_api_key(provided_key, config):
        _log_event(
            "api_auth_failed",
            request_id=request.state.request_id,
            path=request.url.path,
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _extract_api_key(request: Request) -> str | None:
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return api_key

    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return None


def _is_valid_api_key(provided_key: str, config: AppConfig) -> bool:
    provided_hash = _hash_api_key(provided_key)
    for expected_hash in config.api_security.api_key_hashes:
        if compare_digest(provided_hash, expected_hash):
            return True

    for expected_key in _api_keys_from_env(config.api_security.api_key_env_var):
        if compare_digest(provided_key, expected_key):
            return True
    return False


def _hash_api_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()


def _api_keys_from_env(env_var: str) -> tuple[str, ...]:
    raw_value = os.environ.get(env_var, "")
    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def main() -> None:
    import uvicorn

    uvicorn.run("enterprise_rag.api.app:app", host="0.0.0.0", port=8000)
