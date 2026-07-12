from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from enterprise_rag.cache.factory import create_cache
from enterprise_rag.cache.query import build_query_cache_key
from enterprise_rag.config import ApiKeyCredential, AppConfig, load_config, load_config_from_env
from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.embeddings.factory import create_embedding_model
from enterprise_rag.evaluation.ab_testing import ExperimentAssigner, ExperimentVariant, assignment_key
from enterprise_rag.evaluation.readiness import ReadinessReport, build_readiness_report
from enterprise_rag.ingestion.pipeline import IngestReport
from enterprise_rag.jobs.ingest_jobs import IngestJobRecord, IngestJobStore, InMemoryIngestJobStore
from enterprise_rag.jobs.queue import FastApiBackgroundTaskQueue, IngestJobQueue
from enterprise_rag.jobs.runner import IngestJobRunner
from enterprise_rag.leases.factory import create_lease_store
from enterprise_rag.llm.factory import create_llm_client
from enterprise_rag.models import RagAnswer, SearchHit
from enterprise_rag.observability.audit import AuditEvent, AuditLogger, JsonAuditLogger, NullAuditLogger
from enterprise_rag.observability.costs import LLMCostEstimator
from enterprise_rag.observability.feedback import FeedbackRecord, JsonFeedbackStore
from enterprise_rag.observability.tracing import QueryTrace, TraceHit
from enterprise_rag.rag.answer_generation import AnswerGenerator, DeterministicAnswerGenerator, LLMAnswerGenerator
from enterprise_rag.rag.guardrails import QueryGuardrailDecision, QueryGuardrailPolicy
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.rag.query_security import QueryGuard
from enterprise_rag.storage.index_version import IndexVersion, JsonIndexVersionStore
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.base import VectorIndex, VectorSearchResult
from enterprise_rag.vector_index.factory import create_vector_index

DEFAULT_INDEX = Path("data/processed/chunks.json")
DEFAULT_FEEDBACK = Path("data/feedback/feedback.jsonl")
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
TENANT_ID_HEADER = "X-Tenant-ID"
EXPERIMENT_NAME_HEADER = "X-Experiment-Name"
EXPERIMENT_VARIANT_HEADER = "X-Experiment-Variant"
EXPERIMENT_KEY_HEADER = "X-Experiment-Key"
logger = logging.getLogger("enterprise_rag.api")


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 4)


@dataclass(frozen=True)
class AuthContext:
    matched_credential: ApiKeyCredential | None = None


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class ObservedEmbeddingModel:
    def __init__(self, inner: EmbeddingModel, metrics: MetricsCollector, provider: str) -> None:
        self.inner = inner
        self.metrics = metrics
        self.provider = provider

    def embed(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        try:
            return self.inner.embed(text)
        finally:
            self.metrics.record_provider_latency("embedding", self.provider, _elapsed_ms(started_at))


class ObservedVectorIndex:
    def __init__(self, inner: VectorIndex, metrics: MetricsCollector, provider: str) -> None:
        self.inner = inner
        self.metrics = metrics
        self.provider = provider

    def add(self, id: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        self.inner.add(id, vector, metadata=metadata)

    def delete(self, ids: list[str]) -> None:
        self.inner.delete(ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        started_at = time.perf_counter()
        try:
            return self.inner.search(query_vector, top_k=top_k, metadata_filters=metadata_filters)
        finally:
            self.metrics.record_provider_latency("vector_search", self.provider, _elapsed_ms(started_at))


class ObservedAnswerGenerator:
    def __init__(self, inner: AnswerGenerator, metrics: MetricsCollector, provider: str) -> None:
        self.inner = inner
        self.metrics = metrics
        self.provider = provider

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        started_at = time.perf_counter()
        try:
            return self.inner.generate(query, hits)
        finally:
            self.metrics.record_provider_latency("llm_generation", self.provider, _elapsed_ms(started_at))


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
        self.query_estimated_input_cost_usd_sum = 0.0
        self.query_estimated_output_cost_usd_sum = 0.0
        self.query_cache_hits_total = 0
        self.query_cache_misses_total = 0
        self.query_stage_latency_ms_sum: dict[str, float] = {}
        self.query_stage_latency_ms_count: dict[str, int] = {}
        self.provider_latency_ms_sum: dict[tuple[str, str], float] = {}
        self.provider_latency_ms_count: dict[tuple[str, str], int] = {}
        self.ingest_jobs_total = 0
        self.ingest_job_success_total = 0
        self.ingest_job_failures_total = 0
        self.ingest_job_skipped_total = 0
        self.ingest_job_skips_by_reason: dict[str, int] = {}
        self.ingest_job_retry_exhausted_total = 0
        self.ingest_job_latency_ms_sum = 0.0
        self.ingest_job_latency_ms_count = 0
        self.lease_acquire_success_total = 0
        self.lease_acquire_failures_total = 0
        self.feedback_total = 0

    def record_http_request(self) -> None:
        self.http_requests_total += 1

    def record_query_success(self, latency_ms: float, citation_count: int) -> None:
        self.query_requests_total += 1
        self.query_latency_ms_sum += latency_ms
        self.query_latency_ms_count += 1
        self.query_citations_total += citation_count

    def record_query_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        input_cost_usd: float = 0.0,
        output_cost_usd: float = 0.0,
    ) -> None:
        self.query_estimated_input_tokens_total += input_tokens
        self.query_estimated_output_tokens_total += output_tokens
        self.query_estimated_cost_usd_sum += estimated_cost_usd
        self.query_estimated_input_cost_usd_sum += input_cost_usd
        self.query_estimated_output_cost_usd_sum += output_cost_usd

    def record_query_failure(self) -> None:
        self.query_requests_total += 1
        self.query_failures_total += 1

    def record_query_cache_hit(self) -> None:
        self.query_cache_hits_total += 1

    def record_query_cache_miss(self) -> None:
        self.query_cache_misses_total += 1

    def record_query_stage_timings(self, timings_ms: dict[str, float]) -> None:
        for stage, latency_ms in timings_ms.items():
            self.query_stage_latency_ms_sum[stage] = self.query_stage_latency_ms_sum.get(stage, 0.0) + latency_ms
            self.query_stage_latency_ms_count[stage] = self.query_stage_latency_ms_count.get(stage, 0) + 1

    def record_provider_latency(self, component: str, provider: str, latency_ms: float) -> None:
        key = (_metric_suffix(component), _metric_suffix(provider))
        self.provider_latency_ms_sum[key] = self.provider_latency_ms_sum.get(key, 0.0) + latency_ms
        self.provider_latency_ms_count[key] = self.provider_latency_ms_count.get(key, 0) + 1

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
        self.ingest_job_skips_by_reason[reason] = self.ingest_job_skips_by_reason.get(reason, 0) + 1

    def record_ingest_job_retry_exhausted(self) -> None:
        self.ingest_job_retry_exhausted_total += 1

    def record_lease_acquire_success(self) -> None:
        self.lease_acquire_success_total += 1

    def record_lease_acquire_failure(self) -> None:
        self.lease_acquire_failures_total += 1

    def record_feedback(self) -> None:
        self.feedback_total += 1

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
            "enterprise_rag_query_estimated_input_cost_usd_sum": round(self.query_estimated_input_cost_usd_sum, 8),
            "enterprise_rag_query_estimated_output_cost_usd_sum": round(self.query_estimated_output_cost_usd_sum, 8),
            "enterprise_rag_query_cache_hits_total": self.query_cache_hits_total,
            "enterprise_rag_query_cache_misses_total": self.query_cache_misses_total,
            "enterprise_rag_ingest_jobs_total": self.ingest_jobs_total,
            "enterprise_rag_ingest_job_success_total": self.ingest_job_success_total,
            "enterprise_rag_ingest_job_failures_total": self.ingest_job_failures_total,
            "enterprise_rag_ingest_job_skipped_total": self.ingest_job_skipped_total,
            "enterprise_rag_ingest_job_retry_exhausted_total": self.ingest_job_retry_exhausted_total,
            "enterprise_rag_ingest_job_latency_ms_sum": round(self.ingest_job_latency_ms_sum, 4),
            "enterprise_rag_ingest_job_latency_ms_count": self.ingest_job_latency_ms_count,
            "enterprise_rag_lease_acquire_success_total": self.lease_acquire_success_total,
            "enterprise_rag_lease_acquire_failures_total": self.lease_acquire_failures_total,
            "enterprise_rag_feedback_total": self.feedback_total,
        }
        for reason, count in sorted(self.ingest_job_skips_by_reason.items()):
            metrics[f"enterprise_rag_ingest_job_skipped_reason_{_metric_suffix(reason)}_total"] = count
        for stage, latency_sum in sorted(self.query_stage_latency_ms_sum.items()):
            suffix = _metric_suffix(stage)
            metrics[f"enterprise_rag_query_stage_{suffix}_latency_ms_sum"] = round(latency_sum, 4)
            metrics[f"enterprise_rag_query_stage_{suffix}_latency_ms_count"] = self.query_stage_latency_ms_count[stage]
        lines = [f"{name} {value}" for name, value in metrics.items()]
        for (component, provider), latency_sum in sorted(self.provider_latency_ms_sum.items()):
            labels = f'{{component="{component}",provider="{provider}"}}'
            count = self.provider_latency_ms_count[(component, provider)]
            lines.append(f"enterprise_rag_provider_latency_ms_sum{labels} {round(latency_sum, 4)}")
            lines.append(f"enterprise_rag_provider_latency_ms_count{labels} {count}")
            lines.append(f"enterprise_rag_provider_calls_total{labels} {count}")
        return "\n".join(lines) + "\n"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    user_groups: list[str] = Field(default_factory=list)
    include_trace: bool = False


class IngestJobRequest(BaseModel):
    source_path: str = Field(min_length=1)
    sync_vectors: bool = False
    dry_run: bool = False
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
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieved: list[TraceHitResponse]
    reranked: list[TraceHitResponse]
    blocked_context: list[TraceHitResponse]
    final_context: list[TraceHitResponse]


class CostResponse(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class QueryGuardrailResponse(BaseModel):
    needs_human_review: bool = False
    reasons: list[str] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    name: str
    variant: str
    assignment_key: str | None = None
    retrieval_profile: dict[str, object] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    request_id: str
    tenant_id: str | None = None
    index_version: str
    experiment: ExperimentResponse | None = None
    answer: str
    query_plan: QueryPlanResponse
    citations: list[CitationResponse]
    latency_ms: float = 0.0
    cost: CostResponse = Field(default_factory=CostResponse)
    guardrails: QueryGuardrailResponse = Field(default_factory=QueryGuardrailResponse)
    trace: QueryTraceResponse | None = None


@dataclass(frozen=True)
class QueryRuntimeSettings:
    top_k: int
    enable_graph: bool
    graph_max_hops: int


class FeedbackRequest(BaseModel):
    query_request_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rating: str = Field(pattern="^(positive|negative|neutral)$")
    citation_chunk_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    comment: str = ""
    user_id: str | None = None


class FeedbackResponse(BaseModel):
    request_id: str
    feedback_id: str
    status: str = "recorded"


class HealthResponse(BaseModel):
    request_id: str
    status: str
    chunk_count: int
    vector_index_provider: str


class ReadinessCheckResponse(BaseModel):
    name: str
    status: str
    detail: str


class ReadinessResponse(BaseModel):
    request_id: str
    index_present: bool
    chunk_count: int
    eval_present: bool
    eval_case_count: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    query_log_present: bool
    self_healing_draft_present: bool
    self_healing_suggestions_present: bool
    enterprise_checks: list[ReadinessCheckResponse]
    recommendations: list[str]


class IngestReportResponse(BaseModel):
    documents_loaded: int
    documents_new: int
    documents_updated: int
    documents_unchanged: int
    documents_deleted: int
    documents_filtered: int
    filter_reasons: dict[str, int]
    redaction_counts: dict[str, int]
    chunks_indexed: int
    chunks_upserted: list[str]
    chunks_deleted: list[str]
    index_version: str | None = None
    filtered_documents: list[dict[str, str]]
    dry_run: bool


class IngestJobResponse(BaseModel):
    request_id: str
    job_id: str
    status: str
    source_path: str
    tenant_id: str | None = None
    allowed_groups: list[str]
    sync_vectors: bool
    dry_run: bool
    attempt_count: int
    max_attempts: int
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    report: IngestReportResponse | None = None
    vector_sync: dict[str, int] | None = None
    error: str | None = None


class IndexVersionResponse(BaseModel):
    version_id: str
    sequence: int
    updated_at: str
    reason: str
    snapshot_path: str = ""


class IndexRollbackRequest(BaseModel):
    version_id: str = Field(min_length=1)


class IndexRollbackResponse(BaseModel):
    request_id: str
    restored_from_version_id: str
    active_version: IndexVersionResponse


class OpsIndexStatusResponse(BaseModel):
    path: str
    exists: bool
    chunk_count: int
    current_version: str
    version_count: int


class OpsJobsStatusResponse(BaseModel):
    total: int
    by_status: dict[str, int]


class OpsCacheStatusResponse(BaseModel):
    provider: str
    hits: int | None = None
    misses: int | None = None
    entries: int | None = None


class OpsProvidersStatusResponse(BaseModel):
    vector_index: str
    llm: str
    embedding: str
    cache: str
    leases: str


class OpsSecurityStatusResponse(BaseModel):
    api_key_required: bool
    rate_limit_requests: int
    rate_limit_window_seconds: int


class OpsStatusResponse(BaseModel):
    request_id: str
    index: OpsIndexStatusResponse
    jobs: OpsJobsStatusResponse
    query_cache: OpsCacheStatusResponse
    embedding_cache: OpsCacheStatusResponse
    providers: OpsProvidersStatusResponse
    security: OpsSecurityStatusResponse


class FeedbackSummaryResponse(BaseModel):
    total: int
    by_rating: dict[str, int]
    by_label: dict[str, int]


def _log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def _metric_suffix(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "unknown"


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
    app.state.index_version_store = JsonIndexVersionStore(index_path.with_name("index_version.json"))
    app.state.config = config
    app.state.metrics = MetricsCollector()
    app.state.feedback_store = JsonFeedbackStore(index_path.with_name(DEFAULT_FEEDBACK.name))
    app.state.query_guard = QueryGuard()
    app.state.rate_limiter = FixedWindowRateLimiter()
    app.state.embedding_cache = create_cache(config.cache)
    app.state.query_cache = create_cache(config.cache)
    app.state.lease_store = create_lease_store(config.leases)
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
        record_lease_acquire_success=app.state.metrics.record_lease_acquire_success,
        record_lease_acquire_failure=app.state.metrics.record_lease_acquire_failure,
        log_event=_log_event,
        embedding_cache=app.state.embedding_cache,
        embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
        lease_store=app.state.lease_store,
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

    @app.get("/readiness", response_model=ReadinessResponse)
    def readiness(
        request: Request,
        eval_path: str | None = None,
        query_log_path: str | None = None,
        self_healing_dir: str | None = None,
        k: int = Query(default=5, ge=1, le=50),
    ) -> ReadinessResponse:
        _authorize_request(request, app.state.config)
        chunks = JsonChunkStore(app.state.index_path).load()
        report = build_readiness_report(
            chunks,
            index_path=app.state.index_path,
            eval_path=Path(eval_path) if eval_path is not None else None,
            query_log_path=Path(query_log_path) if query_log_path is not None else None,
            self_healing_dir=Path(self_healing_dir) if self_healing_dir is not None else None,
            config=app.state.config,
            k=k,
        )
        return _readiness_response(request.state.request_id, report)

    @app.get("/admin/ops/status", response_model=OpsStatusResponse)
    def ops_status(request: Request) -> OpsStatusResponse:
        _authorize_request(request, app.state.config)
        chunks = JsonChunkStore(app.state.index_path).load()
        versions = app.state.index_version_store.history()
        jobs = app.state.ingest_jobs.list()
        return OpsStatusResponse(
            request_id=request.state.request_id,
            index=OpsIndexStatusResponse(
                path=str(app.state.index_path),
                exists=app.state.index_path.exists(),
                chunk_count=len(chunks),
                current_version=app.state.index_version_store.current_id(app.state.index_path),
                version_count=len(versions),
            ),
            jobs=OpsJobsStatusResponse(total=len(jobs), by_status=_job_status_counts(jobs)),
            query_cache=_cache_status(app.state.config.cache.provider, app.state.query_cache),
            embedding_cache=_cache_status(app.state.config.cache.provider, app.state.embedding_cache),
            providers=OpsProvidersStatusResponse(
                vector_index=app.state.config.vector_index.provider,
                llm=app.state.config.llm.provider,
                embedding=app.state.config.embedding.provider,
                cache=app.state.config.cache.provider,
                leases=app.state.config.leases.provider,
            ),
            security=OpsSecurityStatusResponse(
                api_key_required=app.state.config.api_security.require_api_key,
                rate_limit_requests=app.state.config.api_security.rate_limit_requests,
                rate_limit_window_seconds=app.state.config.api_security.rate_limit_window_seconds,
            ),
        )

    @app.get("/admin/feedback/summary", response_model=FeedbackSummaryResponse)
    def feedback_summary(request: Request) -> FeedbackSummaryResponse:
        _authorize_request(request, app.state.config)
        return _feedback_summary(app.state.feedback_store.load())

    @app.get("/admin/index/versions", response_model=list[IndexVersionResponse])
    def list_index_versions(request: Request) -> list[IndexVersionResponse]:
        _authorize_request(request, app.state.config)
        return [_index_version_response(version) for version in app.state.index_version_store.history()]

    @app.post("/admin/index/rollback", response_model=IndexRollbackResponse)
    def rollback_index(payload: IndexRollbackRequest, request: Request) -> IndexRollbackResponse:
        _authorize_request(request, app.state.config)
        try:
            version = app.state.index_version_store.rollback(
                version_id=payload.version_id,
                index_path=app.state.index_path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _log_event(
            "index_rollback_completed",
            request_id=request.state.request_id,
            restored_from_version_id=payload.version_id,
            active_version_id=version.version_id,
        )
        return IndexRollbackResponse(
            request_id=request.state.request_id,
            restored_from_version_id=payload.version_id,
            active_version=_index_version_response(version),
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
        experiment = _experiment_response(request, config, tenant_id, payload.query)
        runtime = _query_runtime_settings(payload, config, experiment)
        user_groups = set(payload.user_groups) or set(config.security.default_user_groups)
        mandatory_metadata_filters = _tenant_metadata_filter(tenant_id)
        index_version = app.state.index_version_store.current_id(app.state.index_path)
        query_cache_key = build_query_cache_key(
            query=payload.query,
            tenant_id=tenant_id,
            user_groups=user_groups,
            metadata_filters=mandatory_metadata_filters,
            top_k=runtime.top_k,
            index_path=app.state.index_path,
            index_version_id=index_version,
            retrieval_profile=_query_cache_retrieval_profile(config, runtime, experiment),
        )
        cached_response = app.state.query_cache.get(query_cache_key)
        if isinstance(cached_response, dict):
            app.state.metrics.record_query_cache_hit()
            _log_event(
                "query_cache_hit",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                top_k=runtime.top_k,
                index_version=index_version,
            )
            return QueryResponse.model_validate(
                {
                    **cached_response,
                    "request_id": request.state.request_id,
                    "tenant_id": tenant_id,
                    "experiment": experiment.model_dump() if experiment is not None else None,
                    "trace": cached_response.get("trace") if payload.include_trace else None,
                }
            )
        app.state.metrics.record_query_cache_miss()
        answer_generator = ObservedAnswerGenerator(
            _answer_generator_for_config(config) or DeterministicAnswerGenerator(),
            metrics=app.state.metrics,
            provider=config.llm.provider,
        )
        pipeline = RagPipeline(
            chunks,
            enable_graph=runtime.enable_graph,
            graph_max_hops=runtime.graph_max_hops,
            vector_index=ObservedVectorIndex(
                create_vector_index(config.vector_index),
                metrics=app.state.metrics,
                provider=config.vector_index.provider,
            ),
            embedding_model=ObservedEmbeddingModel(
                create_embedding_model(config.embedding),
                metrics=app.state.metrics,
                provider=config.embedding.provider,
            ),
            embedding_cache=app.state.embedding_cache,
            embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
            answer_generator=answer_generator,
        )
        answer, trace = pipeline.answer_for_user_with_trace(
            payload.query,
            top_k=runtime.top_k,
            user_groups=user_groups,
            mandatory_metadata_filters=mandatory_metadata_filters,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        app.state.metrics.record_query_success(latency_ms=latency_ms, citation_count=len(answer.citations))
        app.state.metrics.record_query_stage_timings(trace.timings_ms)
        cost = LLMCostEstimator(config.llm).estimate(
            prompt=_cost_prompt_approximation(payload.query, answer.citations),
            completion=answer.answer,
        )
        app.state.metrics.record_query_cost(
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
            input_cost_usd=_input_cost_usd(config, cost.input_tokens),
            output_cost_usd=_output_cost_usd(config, cost.output_tokens),
        )
        guardrails = QueryGuardrailPolicy(config.guardrails).evaluate(
            payload.query,
            answer.citations,
            cost,
            latency_ms,
        )
        _log_event(
            "query_completed",
            request_id=request.state.request_id,
            latency_ms=round(latency_ms, 2),
            top_k=runtime.top_k,
            enable_graph=runtime.enable_graph,
            graph_max_hops=runtime.graph_max_hops,
            citation_count=len(answer.citations),
            blocked_context_count=len(trace.blocked_context),
            include_trace=payload.include_trace,
            vector_index_provider=config.vector_index.provider,
            index_version=index_version,
            tenant_id=tenant_id,
            experiment=experiment.model_dump() if experiment is not None else None,
            estimated_input_tokens=cost.input_tokens,
            estimated_output_tokens=cost.output_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
            query_stage_timings_ms=trace.timings_ms,
            needs_human_review=guardrails.needs_human_review,
            human_review_reasons=list(guardrails.reasons),
        )
        app.state.audit_logger.log(
            AuditEvent(
                event_type="query.completed",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                principal=_audit_principal(request, auth_context),
                attributes={
                    "top_k": runtime.top_k,
                    "enable_graph": runtime.enable_graph,
                    "graph_max_hops": runtime.graph_max_hops,
                    "query_length": len(payload.query),
                    "citation_chunk_ids": [hit.chunk.id for hit in answer.citations],
                    "index_version": index_version,
                    "experiment": experiment.model_dump() if experiment is not None else None,
                    "user_groups": payload.user_groups,
                    "estimated_input_tokens": cost.input_tokens,
                    "estimated_output_tokens": cost.output_tokens,
                    "estimated_cost_usd": cost.estimated_cost_usd,
                    "needs_human_review": guardrails.needs_human_review,
                    "human_review_reasons": list(guardrails.reasons),
                },
            )
        )
        response = _query_response(
            request.state.request_id,
            tenant_id,
            answer,
            trace if payload.include_trace else None,
            index_version=index_version,
            experiment=experiment,
            latency_ms=latency_ms,
            cost=cost,
            guardrails=guardrails,
        )
        cache_response = _query_response(
            request.state.request_id,
            tenant_id,
            answer,
            trace,
            index_version=index_version,
            experiment=experiment,
            latency_ms=latency_ms,
            cost=cost,
            guardrails=guardrails,
        )
        app.state.query_cache.set(
            query_cache_key, cache_response.model_dump(), ttl_seconds=config.cache.query_ttl_seconds
        )
        return response

    @app.post("/query/stream")
    def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
        response = query(payload, request)
        return StreamingResponse(_stream_query_response(response), media_type="application/x-ndjson")

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def submit_feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        feedback_id = f"fb_{uuid4().hex}"
        record = FeedbackRecord(
            feedback_id=feedback_id,
            request_id=payload.query_request_id,
            query=payload.query,
            answer=payload.answer,
            rating=payload.rating,
            tenant_id=tenant_id,
            user_id=payload.user_id,
            citation_chunk_ids=tuple(payload.citation_chunk_ids),
            labels=tuple(payload.labels),
            comment=payload.comment,
        )
        app.state.feedback_store.append(record)
        app.state.metrics.record_feedback()
        app.state.audit_logger.log(
            AuditEvent(
                event_type="feedback.recorded",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                principal=_audit_principal(request, auth_context),
                attributes={
                    "feedback_id": feedback_id,
                    "query_request_id": payload.query_request_id,
                    "rating": payload.rating,
                    "labels": payload.labels,
                    "citation_chunk_ids": payload.citation_chunk_ids,
                },
            )
        )
        _log_event(
            "feedback_recorded",
            request_id=request.state.request_id,
            feedback_id=feedback_id,
            query_request_id=payload.query_request_id,
            tenant_id=tenant_id,
            rating=payload.rating,
            labels=payload.labels,
        )
        return FeedbackResponse(request_id=request.state.request_id, feedback_id=feedback_id)

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
            dry_run=payload.dry_run,
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
                    "dry_run": payload.dry_run,
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
            dry_run=payload.dry_run,
        )
        return _ingest_job_response(request.state.request_id, job)

    @app.get("/ingest-jobs", response_model=list[IngestJobResponse])
    def list_ingest_jobs(
        request: Request,
        status: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> list[IngestJobResponse]:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        jobs = app.state.ingest_jobs.list()
        if tenant_id is not None:
            jobs = [job for job in jobs if job.tenant_id == tenant_id]
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        jobs = sorted(jobs, key=lambda job: job.created_at, reverse=True)[:limit]
        return [_ingest_job_response(request.state.request_id, job) for job in jobs]

    @app.get("/ingest-jobs/{job_id}", response_model=IngestJobResponse)
    def get_ingest_job(job_id: str, request: Request) -> IngestJobResponse:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        job = _get_tenant_scoped_ingest_job(app.state.ingest_jobs, job_id, tenant_id)
        return _ingest_job_response(request.state.request_id, job)

    @app.post("/ingest-jobs/{job_id}/cancel", response_model=IngestJobResponse)
    def cancel_ingest_job(job_id: str, request: Request) -> IngestJobResponse:
        auth_context = _authorize_request(request, app.state.config)
        tenant_id = _resolve_tenant_id(request, app.state.config, auth_context)
        job = _get_tenant_scoped_ingest_job(app.state.ingest_jobs, job_id, tenant_id)
        if job.status not in {"queued", "failed"}:
            raise HTTPException(status_code=409, detail=f"Cannot cancel ingest job with status `{job.status}`.")
        app.state.ingest_jobs.mark_canceled(job_id)
        canceled = app.state.ingest_jobs.get(job_id)
        if canceled is None:
            raise HTTPException(status_code=404, detail="Ingest job not found.")
        app.state.audit_logger.log(
            AuditEvent(
                event_type="ingest_job.canceled",
                request_id=request.state.request_id,
                tenant_id=tenant_id,
                principal=_audit_principal(request, auth_context),
                attributes={"job_id": job_id},
            )
        )
        _log_event(
            "ingest_job_canceled",
            request_id=request.state.request_id,
            job_id=job_id,
            tenant_id=tenant_id,
        )
        return _ingest_job_response(request.state.request_id, canceled)

    return app


app = create_app(config=load_config_from_env())


def _answer_generator_for_config(config: AppConfig) -> LLMAnswerGenerator | None:
    if config.llm.provider.lower() == "stub":
        return None
    return LLMAnswerGenerator(create_llm_client(config.llm))


def _get_tenant_scoped_ingest_job(
    job_store: IngestJobStore,
    job_id: str,
    tenant_id: str | None,
) -> IngestJobRecord:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    if tenant_id is not None and job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return job


def _query_response(
    request_id: str,
    tenant_id: str | None,
    answer: RagAnswer,
    trace: QueryTrace | None,
    index_version: str,
    experiment: ExperimentResponse | None = None,
    latency_ms: float = 0.0,
    cost: object | None = None,
    guardrails: QueryGuardrailDecision | None = None,
) -> QueryResponse:
    return QueryResponse(
        request_id=request_id,
        tenant_id=tenant_id,
        index_version=index_version,
        experiment=experiment,
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
        latency_ms=round(latency_ms, 2),
        cost=_cost_response(cost),
        guardrails=_guardrail_response(guardrails),
        trace=_trace_response(trace) if trace is not None else None,
    )


def _stream_query_response(response: QueryResponse) -> Iterable[str]:
    yield _json_line(
        {
            "event": "metadata",
            "request_id": response.request_id,
            "tenant_id": response.tenant_id,
            "index_version": response.index_version,
            "experiment": response.experiment.model_dump() if response.experiment is not None else None,
            "latency_ms": response.latency_ms,
            "cost": response.cost.model_dump(),
            "guardrails": response.guardrails.model_dump(),
        }
    )
    for token in response.answer.split():
        yield _json_line({"event": "answer_delta", "text": f"{token} "})
    yield _json_line({"event": "citations", "citations": [citation.model_dump() for citation in response.citations]})
    if response.trace is not None:
        yield _json_line({"event": "trace", "trace": response.trace.model_dump()})
    yield _json_line({"event": "done"})


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True) + "\n"


def _job_status_counts(jobs: list[IngestJobRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    return counts


def _cache_status(provider: str, cache: object) -> OpsCacheStatusResponse:
    entries = getattr(cache, "entries", None)
    return OpsCacheStatusResponse(
        provider=provider,
        hits=getattr(cache, "hits", None),
        misses=getattr(cache, "misses", None),
        entries=len(entries) if isinstance(entries, dict) else None,
    )


def _experiment_response(
    request: Request,
    config: AppConfig,
    tenant_id: str | None,
    query: str,
) -> ExperimentResponse | None:
    name = request.headers.get(EXPERIMENT_NAME_HEADER, "").strip()
    variant = request.headers.get(EXPERIMENT_VARIANT_HEADER, "").strip()
    manual_assignment_key = request.headers.get(EXPERIMENT_KEY_HEADER)
    if name or variant:
        return ExperimentResponse(
            name=name or "manual",
            variant=variant or "unspecified",
            assignment_key=manual_assignment_key,
        )
    if not config.experiments.enabled or not config.experiments.variants:
        return None
    assigner = ExperimentAssigner(
        config.experiments.name,
        tuple(
            ExperimentVariant(
                name=variant_config.name,
                traffic_weight=variant_config.traffic_weight,
                retrieval_profile=variant_config.retrieval_profile,
            )
            for variant_config in config.experiments.variants
        ),
    )
    key = manual_assignment_key or assignment_key(
        tenant_id=tenant_id,
        user_id=None,
        query=query,
    )
    assignment = assigner.assign(key)
    return ExperimentResponse(
        name=assignment.experiment_name,
        variant=assignment.variant_name,
        assignment_key=assignment.assignment_key,
        retrieval_profile=assignment.retrieval_profile,
    )


def _query_runtime_settings(
    payload: QueryRequest,
    config: AppConfig,
    experiment: ExperimentResponse | None,
) -> QueryRuntimeSettings:
    profile = experiment.retrieval_profile if experiment is not None else {}
    top_k = payload.top_k if payload.top_k is not None else _profile_int(profile, "top_k", config.retrieval.top_k)
    return QueryRuntimeSettings(
        top_k=top_k,
        enable_graph=_profile_bool(profile, "enable_graph", config.retrieval.enable_graph),
        graph_max_hops=_profile_int(profile, "graph_max_hops", config.retrieval.graph_max_hops),
    )


def _profile_int(profile: dict[str, object], key: str, default: int) -> int:
    value = profile.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=500, detail=f"Experiment retrieval_profile `{key}` must be an integer.")
    return value


def _profile_bool(profile: dict[str, object], key: str, default: bool) -> bool:
    value = profile.get(key, default)
    if not isinstance(value, bool):
        raise HTTPException(status_code=500, detail=f"Experiment retrieval_profile `{key}` must be a boolean.")
    return value


def _feedback_summary(records: list[FeedbackRecord]) -> FeedbackSummaryResponse:
    by_rating: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for record in records:
        by_rating[record.rating] = by_rating.get(record.rating, 0) + 1
        for label in record.labels:
            by_label[label] = by_label.get(label, 0) + 1
    return FeedbackSummaryResponse(total=len(records), by_rating=by_rating, by_label=by_label)


def _cost_response(cost: object | None) -> CostResponse:
    if cost is None:
        return CostResponse()
    return CostResponse(
        input_tokens=int(getattr(cost, "input_tokens", 0)),
        output_tokens=int(getattr(cost, "output_tokens", 0)),
        estimated_cost_usd=float(getattr(cost, "estimated_cost_usd", 0.0)),
    )


def _guardrail_response(guardrails: QueryGuardrailDecision | None) -> QueryGuardrailResponse:
    if guardrails is None:
        return QueryGuardrailResponse()
    return QueryGuardrailResponse(
        needs_human_review=guardrails.needs_human_review,
        reasons=list(guardrails.reasons),
    )


def _readiness_response(request_id: str, report: ReadinessReport) -> ReadinessResponse:
    return ReadinessResponse(
        request_id=request_id,
        index_present=report.index_present,
        chunk_count=report.chunk_count,
        eval_present=report.eval_present,
        eval_case_count=report.eval_case_count,
        recall_at_k=report.recall_at_k,
        precision_at_k=report.precision_at_k,
        mrr=report.mrr,
        query_log_present=report.query_log_present,
        self_healing_draft_present=report.self_healing_draft_present,
        self_healing_suggestions_present=report.self_healing_suggestions_present,
        enterprise_checks=[
            ReadinessCheckResponse(name=check.name, status=check.status, detail=check.detail)
            for check in report.enterprise_checks
        ],
        recommendations=list(report.recommendations),
    )


def _query_cache_retrieval_profile(
    config: AppConfig,
    runtime: QueryRuntimeSettings,
    experiment: ExperimentResponse | None = None,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "retrieval": {
            "enable_graph": runtime.enable_graph,
            "graph_max_hops": runtime.graph_max_hops,
        },
        "vector_index": {
            "provider": config.vector_index.provider,
            "collection_name": config.vector_index.collection_name,
            "url": config.vector_index.url,
        },
        "embedding": {
            "model_id": "hashing-embedding-v1",
        },
        "reranker": {
            "profile": "lightweight-reranker-v1",
        },
        "answer_generator": {
            "provider": config.llm.provider,
            "model": config.llm.model,
        },
    }
    if experiment is not None:
        profile["experiment"] = experiment.model_dump()
    return profile


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
        timings_ms=trace.timings_ms,
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


def _index_version_response(version: IndexVersion) -> IndexVersionResponse:
    return IndexVersionResponse(
        version_id=version.version_id,
        sequence=version.sequence,
        updated_at=version.updated_at,
        reason=version.reason,
        snapshot_path=version.snapshot_path,
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
        dry_run=job.dry_run,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        lease_owner=job.lease_owner,
        lease_expires_at=job.lease_expires_at,
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
        filter_reasons=report.filter_reasons or {},
        redaction_counts=report.redaction_counts or {},
        chunks_indexed=report.chunks_indexed,
        chunks_upserted=list(report.chunks_upserted),
        chunks_deleted=list(report.chunks_deleted),
        index_version=report.index_version,
        filtered_documents=[
            {"source_path": item.source_path, "reason": item.reason} for item in report.filtered_documents
        ],
        dry_run=report.dry_run,
    )


def _cost_prompt_approximation(query: str, citations: tuple[SearchHit, ...]) -> str:
    evidence = "\n".join(hit.chunk.text for hit in citations)
    return f"{query}\n{evidence}"


def _input_cost_usd(config: AppConfig, input_tokens: int) -> float:
    return round(input_tokens / 1000 * config.llm.input_cost_per_1k_tokens, 8)


def _output_cost_usd(config: AppConfig, output_tokens: int) -> float:
    return round(output_tokens / 1000 * config.llm.output_cost_per_1k_tokens, 8)


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
