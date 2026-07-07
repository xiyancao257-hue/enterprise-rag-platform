from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from enterprise_rag.config import AppConfig, load_config
from enterprise_rag.models import RagAnswer, SearchHit
from enterprise_rag.observability.tracing import QueryTrace, TraceHit
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.factory import create_vector_index

DEFAULT_INDEX = Path("data/processed/chunks.json")


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
    answer: str
    query_plan: QueryPlanResponse
    citations: list[CitationResponse]
    trace: QueryTraceResponse | None = None


class HealthResponse(BaseModel):
    status: str
    chunk_count: int
    vector_index_provider: str


def create_app(index_path: Path = DEFAULT_INDEX, config_path: Path | None = None) -> FastAPI:
    config = load_config(config_path)
    app = FastAPI(title="Enterprise RAG API", version="0.1.0")
    app.state.index_path = index_path
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        chunks = JsonChunkStore(app.state.index_path).load()
        return HealthResponse(
            status="ok",
            chunk_count=len(chunks),
            vector_index_provider=app.state.config.vector_index.provider,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        chunks = JsonChunkStore(app.state.index_path).load()
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found. Run ingestion before querying.")

        config: AppConfig = app.state.config
        top_k = request.top_k if request.top_k is not None else config.retrieval.top_k
        pipeline = RagPipeline(
            chunks,
            enable_graph=config.retrieval.enable_graph,
            graph_max_hops=config.retrieval.graph_max_hops,
            vector_index=create_vector_index(config.vector_index),
        )
        answer, trace = pipeline.answer_for_user_with_trace(
            request.query,
            top_k=top_k,
            user_groups=set(request.user_groups) or set(config.security.default_user_groups),
        )
        return _query_response(answer, trace if request.include_trace else None)

    return app


app = create_app()


def _query_response(answer: RagAnswer, trace: QueryTrace | None) -> QueryResponse:
    return QueryResponse(
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


def main() -> None:
    import uvicorn

    uvicorn.run("enterprise_rag.api.app:app", host="0.0.0.0", port=8000)
