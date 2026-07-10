from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from enterprise_rag.config import ApiKeyCredential, AppConfig, load_config
from enterprise_rag.ingestion.pipeline import IngestReport
from enterprise_rag.jobs.ingest_jobs import IngestJobRecord, IngestJobStore, InMemoryIngestJobStore
from enterprise_rag.jobs.queue import FastApiBackgroundTaskQueue, IngestJobQueue
from enterprise_rag.jobs.runner import IngestJobRunner
from enterprise_rag.models import RagAnswer, SearchHit
from enterprise_rag.observability.audit import AuditEvent, AuditLogger, JsonAuditLogger, NullAuditLogger
from enterprise_rag.observability.costs import LLMCostEstimator
from enterprise_rag.observability.tracing import QueryTrace, TraceHit
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.rag.query_security import QueryGuard
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.factory import create_vector_index

DEFAULT_INDEX = Path("data/processed/chunks.json")
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
TENANT_ID_HEADER = "X-Tenant-ID"
logger = logging.getLogger("enterprise_rag.api")


@dataclass(frozen=True)
class AuthContext:
    matched_credential: ApiKeyCredential | None = None


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class FixedWindowRateLimiter:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self.now = now or time.time
        self.windows: dict[str, tuple[float, int]] = {}

    def check(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(allowed=True)

        now = self.now()
        window_start, count = self.windows.get(key, (now, 0))
        if now - window_start >= window_seconds:
            window_start = now
            count = 0

        if count >= limit:
            retry_after = max(1, int(window_seconds - (now - window_start)))
            return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

        self.windows[key] = (window_start, count + 1)
        return RateLimitDecision(allowed=True)


class MetricsCollector:
    def __init__(self) -> None:
        self.http_requests_total = 0
        self.query_requests_total = 0
        self.query_failures_total = 0
        self.query_latency_ms_sum = 0.0
        self.query_latency_ms_count = 0
        self.query_citations_total = 0
        self.query_estimated_input_tokens_total = 0
        self.query_estimated_output_tokens_total = 0
        self.query_estimated_cost_usd_sum = 0.0
        self.ingest_jobs_total = 0
        self.ingest_job_success_total = 0
        self.ingest_job_failures_total = 0
        self.ingest_job_skipped_total = 0
        self.ingest_job_retry_exhausted_total = 0
        self.ingest_job_latency_ms_sum = 0.0
        self.ingest_job_latency_ms_count = 0

    def record_http_request(self) -> None:
        self.http_requests_total += 1

    def record_query_success(self, latency_ms: float, citation_count: int) -> None:
        self.query_requests_total += 1
        self.query_latency_ms_sum += latency_ms
        self.query_latency_ms_count += 1
        self.query_citations_total += citation_count

    def record_query_cost(self, input_tokens: int, output_tokens: int, estimated_cost_usd: float) -> None:
        self.query_estimated_input_tokens_total += input_tokens
        self.query_estimated_output_tokens_total += output_tokens
        self.query_estimated_cost_usd_sum += estimated_cost_usd

    def record_query_failure(self) -> None:
        self.query_requests_total += 1
        self.query_failures_total += 1

    def record_ingest_job_created(self) -> None:
        self.ingest_jobs_total += 1

    def record_ingest_job_success(self, latency_ms: float) -> None:
        self.ingest_job_success_total += 1
        self.ingest_job_latency_ms_sum += latency_ms
        self.ingest_job_latency_ms_count += 1

    def record_ingest_job_failure(self) -> None:
        self.ingest_job_failures_total += 1

    def record_ingest_job_skip(self, reason: str) -> None:
        self.ingest_job_skipped_total += 1

    def record_ingest_job_retry_exhausted(self) -> None:
        self.ingest_job_retry_exhausted_total += 1

    def render_prometheus(self) -> str:
        metrics = {
            "enterprise_rag_http_requests_total": self.http_requests_total,
            "enterprise_rag_query_requests_total": self.query_requests_total,
            "enterprise_rag_query_failures_total": self.query_failures_total,
            "enterprise_rag_query_latency_ms_sum": round(self.query_latency_ms_sum, 4),
            "enterprise_rag_query_latency_ms_count": self.query_latency_ms_count,
            "enterprise_rag_query_citations_total": self.query_citations_total,
            "enterprise_rag_query_estimated_input_tokens_total": self.query_estimated_input_tokens_total,
            "enterprise_rag_query_estimated_output_tokens_total": self.query_estimated_output_tokens_total,
            "enterprise_rag_query_estimated_cost_usd_sum": round(self.query_estimated_cost_usd_sum, 8),
            "enterprise_rag_ingest_jobs_total": self.ingest_jobs_total,
            "enterprise_rag_ingest_job_success_total": self.ingest_job_success_total,
            "enterprise_rag_ingest_job_failures_total": self.ingest_job_failures_total,
            "enterprise_rag_ingest_job_skipped_total": self.ingest_job_skipped_total,
            "enterprise_rag_ingest_job_retry_exhausted_total": self.ingest_job_retry_exhausted_total,
            "enterprise_rag_ingest_job_latency_ms_sum": round(self.ingest_job_latency_ms_sum, 4),
            "enterprise_rag_ingest_job_latency_ms_count": self.ingest_job_latency_ms_count,
        }
        return "\n".join(f"{name} {value}" for name, value in metrics.items()) + "\n"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    user_groups: list[str] = Field(default_factory=list)
    include_trace: bool = False


class IngestJobRequest(BaseModel):
    source_path: str = Field(min_length=1)
    sync_vectors: bool = False
    allowed_groups: list[str] = Field(default_factory=list)


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
    blocked_context: list[TraceHitResponse]
    final_context: list[TraceHitResponse]


class QueryResponse(BaseModel):
    request_id: str
    tenant_id: str | None = None
    answer: str
    query_plan: QueryPlanResponse
    citations: list[CitationResponse]
    trace: QueryTraceResponse | None = None


class HealthResponse(BaseModel):
    request_id: str
    status: str
    chunk_count: int
    vector_index_provider: str


class IngestReportResponse(BaseModel):
    documents_loaded: int
    documents_new: int
    documents_updated: int
    documents_unchanged: int
    documents_deleted: int
    documents_filtered: int
    chunks_indexed: int
    chunks_upserted: list[str]
    chunks_deleted: list[str]


class IngestJobResponse(BaseModel):
    request_id: str
    job_id: str
    status: str
    source_path: str
    tenant_id: str | None = None
    allowed_groups: list[str]
    sync_vectors: bool
    attempt_count: int
    max_attempts: int
    report: IngestReportResponse | None = None
    vector_sync: dict[str, int] | None = None
    error: str | None = None


def _log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def create_app(
    index_path: Path = DEFAULT_INDEX,
    config_path: Path | None = None,
    config: AppConfig | None = None,
    ingest_job_store: IngestJobStore | None = None,
    ingest_job_queue_factory: Callable[[BackgroundTasks, IngestJobRunner], IngestJobQueue] | None = None,
    audit_logger: AuditLogger | None = None,
) -> FastAPI:
    config = config or load_config(config_path)
    app = FastAPI(title="Enterprise RAG API", version="0.1.0")
    app.state.index_path = index_path
    app.state.config = config
    app.state.metrics = MetricsCollector()
    app.state.query_guard = QueryGuard()
    app.state.rate_limiter = FixedWindowRateLimiter()
    app.state.audit_logger = audit_logger or (
        JsonAuditLogger(Path(config.audit.path)) if config.audit.enabled else NullAuditLogger()
    )
    app.state.ingest_jobs = ingest_job_store or InMemoryIngestJobStore()
    app.state.ingest_job_runner = IngestJobRunner(
        job_store=app.state.ingest_jobs,
        index_path=index_path,
        config=config,
        record_failure=app.state.metrics.record_ingest_job_failure,
        record_success=app.state.metrics.record_ingest_job_success,
        record_skip=app.state.metrics.record_ingest_job_skip,
        record_retry_exhausted=app.state.metrics.record_ingest_job_retry_exhausted,
        log_event=_log_event,
    )
    app.state.ingest_job_queue_factory = ingest_job_queue_factory or FastApiBackgroundTaskQueue

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
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        started_at = time.perf_counter()
        _enforce_rate_limit(request, app.state.config, tenant_id, auth_context, started_at)
        query_security = app.state.query_guard.check(payload.query)
        if not query_security.allowed:
            app.state.metrics.record_query_failure()
            findings = [{"label": finding.label, "message": finding.message} for finding in query_security.findings]
            app.state.audit_logger.log(
                AuditEvent(
                    event_type="query.rejected",
                    request_id=request.state.request_id,
                    tenant_id=tenant_id,
                    principal=_audit_principal(request, auth_context),
                    attributes={
                        "reason": "safety_policy",
                        "findings": findings,
                    },
                )
            )
            _log_event(
                "query_rejected",
                request_id=request.state.request_id,
                reason=";".join(finding.label for finding in query_security.findings),
                tenant_id=tenant_id,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Query rejected by safety policy.",
                    "findings": findings,
                },
            )
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
            mandatory_metadata_filters=_tenant_metadata_filter(tenant_id),
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        app.state.metrics.record_query_success(latency_ms=latency_ms, citation_count=len(answer.citations))
        cost = LLMCostEstimator(config.llm).estimate(
            prompt=_cost_prompt_approximation(payload.query, answer.citations),
            completion=answer.answer,
        )
        app.state.metrics.record_query_cost(
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
        )
        _log_event(
            "query_completed",
            request_id=request.state.request_id,
            latency_ms=round(latency_ms, 2),
            top_k=top_k,
            citation_count=len(answer.citations),
            blocked_context_count=len(trace.blocked_context),
            include_trace=payload.include_trace,
            vector_index_provider=config.vector_index.provider,
            tenant_id=tenant_id,
            estimated_input_tokens=cost.input_tokens,
            estimated_output_tokens=cost.output_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
        )
        app.state.audit_logger.log(
            AuditEvent(
                event_type="query.completed",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                principal=_audit_principal(request, auth_context),
                attributes={
                    "top_k": top_k,
                    "query_length": len(payload.query),
                    "citation_chunk_ids": [hit.chunk.id for hit in answer.citations],
                    "user_groups": payload.user_groups,
                    "estimated_input_tokens": cost.input_tokens,
                    "estimated_output_tokens": cost.output_tokens,
                    "estimated_cost_usd": cost.estimated_cost_usd,
                },
            )
        )
        return _query_response(request.state.request_id, tenant_id, answer, trace if payload.include_trace else None)

    @app.post("/ingest-jobs", response_model=IngestJobResponse, status_code=202)
    def create_ingest_job(
        payload: IngestJobRequest,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> IngestJobResponse:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        source_path = _validate_ingest_source_path(Path(payload.source_path), app.state.config)

        job = app.state.ingest_jobs.create(
            source_path=str(source_path),
            tenant_id=tenant_id,
            allowed_groups=tuple(payload.allowed_groups),
            sync_vectors=payload.sync_vectors,
            request_id=request.state.request_id,
        )
        app.state.metrics.record_ingest_job_created()
        app.state.ingest_job_queue_factory(background_tasks, app.state.ingest_job_runner).publish(job.job_id)
        app.state.audit_logger.log(
            AuditEvent(
                event_type="ingest_job.created",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                principal=_audit_principal(request, auth_context),
                attributes={
                    "job_id": job.job_id,
                    "source_path": str(source_path),
                    "sync_vectors": payload.sync_vectors,
                    "allowed_groups": payload.allowed_groups,
                },
            )
        )
        _log_event(
            "ingest_job_created",
            request_id=request.state.request_id,
            job_id=job.job_id,
            tenant_id=tenant_id,
            sync_vectors=payload.sync_vectors,
        )
        return _ingest_job_response(request.state.request_id, job)

    @app.get("/ingest-jobs/{job_id}", response_model=IngestJobResponse)
    def get_ingest_job(job_id: str, request: Request) -> IngestJobResponse:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        job = app.state.ingest_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found.")
        if tenant_id is not None and job.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Ingest job not found.")
        return _ingest_job_response(request.state.request_id, job)

    return app


app = create_app()


def _query_response(
    request_id: str,
    tenant_id: str | None,
    answer: RagAnswer,
    trace: QueryTrace | None,
) -> QueryResponse:
    return QueryResponse(
        request_id=request_id,
        tenant_id=tenant_id,
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
        blocked_context=[_trace_hit_response(hit) for hit in trace.blocked_context],
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


def _ingest_job_response(request_id: str, job: IngestJobRecord) -> IngestJobResponse:
    return IngestJobResponse(
        request_id=request_id,
        job_id=job.job_id,
        status=job.status,
        source_path=job.source_path,
        tenant_id=job.tenant_id,
        allowed_groups=list(job.allowed_groups),
        sync_vectors=job.sync_vectors,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        report=_ingest_report_response(job.report) if job.report is not None else None,
        vector_sync=job.vector_sync,
        error=job.error,
    )


def _ingest_report_response(report: IngestReport) -> IngestReportResponse:
    return IngestReportResponse(
        documents_loaded=report.documents_loaded,
        documents_new=report.documents_new,
        documents_updated=report.documents_updated,
        documents_unchanged=report.documents_unchanged,
        documents_deleted=report.documents_deleted,
        documents_filtered=report.documents_filtered,
        chunks_indexed=report.chunks_indexed,
        chunks_upserted=list(report.chunks_upserted),
        chunks_deleted=list(report.chunks_deleted),
    )


def _cost_prompt_approximation(query: str, citations: tuple[SearchHit, ...]) -> str:
    evidence = "\n".join(hit.chunk.text for hit in citations)
    return f"{query}\n{evidence}"


def _validate_ingest_source_path(source_path: Path, config: AppConfig) -> Path:
    resolved_source = source_path.resolve(strict=False)
    allowed_roots = tuple(Path(root).resolve(strict=False) for root in config.ingestion.allowed_source_roots)
    if allowed_roots and not any(_is_relative_to(resolved_source, root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Ingest source path is not allowed.")
    if not resolved_source.exists():
        raise HTTPException(status_code=400, detail="Ingest source path does not exist.")
    return resolved_source


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _authorize_request(request: Request, config: AppConfig) -> AuthContext:
    if not config.api_security.require_api_key:
        return AuthContext()

    provided_key = _extract_api_key(request)
    matched_credential = _match_api_key(provided_key, config) if provided_key is not None else None
    if matched_credential is None and (provided_key is None or not _is_valid_api_key(provided_key, config)):
        _log_event(
            "api_auth_failed",
            request_id=request.state.request_id,
            path=request.url.path,
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return AuthContext(matched_credential=matched_credential)


def _resolve_tenant_id(request: Request, config: AppConfig, auth_context: AuthContext) -> str | None:
    tenant_id = request.headers.get(TENANT_ID_HEADER)
    if tenant_id:
        tenant_id = tenant_id.strip()
        _authorize_tenant(request, tenant_id, auth_context)
        return tenant_id
    if config.api_security.require_api_key:
        _log_event(
            "tenant_auth_failed",
            request_id=request.state.request_id,
            path=request.url.path,
        )
        raise HTTPException(status_code=400, detail="Missing X-Tenant-ID header.")
    return None


def _authorize_tenant(request: Request, tenant_id: str, auth_context: AuthContext) -> None:
    credential = auth_context.matched_credential
    if credential is None or not credential.allowed_tenants or "*" in credential.allowed_tenants:
        return
    if tenant_id in credential.allowed_tenants:
        return
    _log_event(
        "tenant_forbidden",
        request_id=request.state.request_id,
        path=request.url.path,
        tenant_id=tenant_id,
    )
    raise HTTPException(status_code=403, detail="API key is not allowed for this tenant.")


def _enforce_rate_limit(
    request: Request,
    config: AppConfig,
    tenant_id: str | None,
    auth_context: AuthContext,
    started_at: float,
) -> None:
    limit_key = _rate_limit_key(request, tenant_id, auth_context)
    decision = request.app.state.rate_limiter.check(
        limit_key,
        limit=config.api_security.rate_limit_requests,
        window_seconds=config.api_security.rate_limit_window_seconds,
    )
    if decision.allowed:
        return

    request.app.state.metrics.record_query_failure()
    _log_event(
        "query_rate_limited",
        request_id=request.state.request_id,
        tenant_id=tenant_id,
        retry_after_seconds=decision.retry_after_seconds,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded.",
        headers={"Retry-After": str(decision.retry_after_seconds)},
    )


def _rate_limit_key(request: Request, tenant_id: str | None, auth_context: AuthContext) -> str:
    credential = auth_context.matched_credential
    if credential is not None:
        principal = credential.key_hash
    else:
        principal = request.headers.get(API_KEY_HEADER) or request.client.host if request.client else "anonymous"
    return f"{principal}:{tenant_id or 'public'}"


def _audit_principal(request: Request, auth_context: AuthContext) -> str:
    credential = auth_context.matched_credential
    if credential is not None:
        return f"api_key:{credential.key_hash}"
    provided_key = _extract_api_key(request)
    if provided_key:
        return f"api_key:{_hash_api_key(provided_key)}"
    return "anonymous"


def _tenant_metadata_filter(tenant_id: str | None) -> dict[str, str]:
    if tenant_id is None:
        return {}
    return {"tenant_id": tenant_id}


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


def _match_api_key(provided_key: str, config: AppConfig) -> ApiKeyCredential | None:
    provided_hash = _hash_api_key(provided_key)
    for credential in config.api_security.api_keys:
        if compare_digest(provided_hash, credential.key_hash):
            return credential
    return None


def _hash_api_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()


def _api_keys_from_env(env_var: str) -> tuple[str, ...]:
    raw_value = os.environ.get(env_var, "")
    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def main() -> None:
    import uvicorn

    uvicorn.run("enterprise_rag.api.app:app", host="0.0.0.0", port=8000)
