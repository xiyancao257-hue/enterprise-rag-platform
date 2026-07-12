from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.config import AppConfig
from enterprise_rag.indexing.vector_sync import VectorIndexSync
from enterprise_rag.ingestion.ocr_factory import create_ocr_adapter
from enterprise_rag.ingestion.pipeline import IncrementalIngestPipeline
from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.jobs.ingest_jobs import IngestJobRecord, IngestJobStore
from enterprise_rag.leases.base import LeaseStore
from enterprise_rag.leases.in_memory import InMemoryLeaseStore
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.factory import create_vector_index


class IngestJobRunner:
    RUNNABLE_STATUSES = {"queued", "failed"}

    def __init__(
        self,
        job_store: IngestJobStore,
        index_path: Path,
        config: AppConfig,
        record_failure: Callable[[], None] | None = None,
        record_success: Callable[[float], None] | None = None,
        record_skip: Callable[[str], None] | None = None,
        record_retry_exhausted: Callable[[], None] | None = None,
        record_lease_acquire_success: Callable[[], None] | None = None,
        record_lease_acquire_failure: Callable[[], None] | None = None,
        log_event: Callable[..., None] | None = None,
        now: Callable[[], float] | None = None,
        worker_id: str | None = None,
        embedding_cache: CacheStore | None = None,
        embedding_ttl_seconds: int | None = None,
        lease_store: LeaseStore | None = None,
    ) -> None:
        self.job_store = job_store
        self.index_path = index_path
        self.config = config
        self.record_failure = record_failure or (lambda: None)
        self.record_success = record_success or (lambda latency_ms: None)
        self.record_skip = record_skip or (lambda reason: None)
        self.record_retry_exhausted = record_retry_exhausted or (lambda: None)
        self.record_lease_acquire_success = record_lease_acquire_success or (lambda: None)
        self.record_lease_acquire_failure = record_lease_acquire_failure or (lambda: None)
        self.log_event = log_event or (lambda event, **fields: None)
        self.now = now or time.time
        self.worker_id = worker_id or f"worker_{uuid4().hex}"
        self.embedding_cache = embedding_cache or InMemoryCache()
        self.embedding_ttl_seconds = (
            embedding_ttl_seconds if embedding_ttl_seconds is not None else config.cache.embedding_ttl_seconds
        )
        self.lease_store = lease_store or InMemoryLeaseStore()

    def run(self, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if job is None:
            return
        if job.status == "running" and not self._is_stale_running(job):
            self._skip_job(job, reason="running")
            return
        if job.status in {"failed", "running"} and job.attempt_count >= job.max_attempts:
            self.record_retry_exhausted()
            self.log_event(
                "ingest_job_retry_exhausted",
                request_id=job.request_id,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                status=job.status,
                attempt_count=job.attempt_count,
                max_attempts=job.max_attempts,
            )
            return
        if job.status not in self.RUNNABLE_STATUSES and job.status != "running":
            self._skip_job(job, reason=job.status)
            return

        lease_name = f"ingest-job:{job_id}"
        if not self.lease_store.acquire(lease_name, self.worker_id, self.config.jobs.running_timeout_seconds):
            self.record_lease_acquire_failure()
            self._skip_job(job, reason="distributed_lease")
            return
        self.record_lease_acquire_success()

        started_at = time.perf_counter()
        try:
            lease_expires_at = self.now() + self.config.jobs.running_timeout_seconds
            if not self.job_store.acquire_lease(job_id, self.worker_id, lease_expires_at):
                leased_job = self.job_store.get(job_id)
                if leased_job is not None:
                    self._skip_job(leased_job, reason="leased")
                return

            job = self.job_store.get(job_id)
            if job is None:
                return
            metadata_overrides = self._metadata_overrides(job)
            store = JsonChunkStore(self.index_path)
            report = IncrementalIngestPipeline(
                file_policy=IngestionFilePolicy.from_config(self.config.ingestion),
                ocr_adapter=create_ocr_adapter(self.config.ocr),
                chunking_config=self.config.chunking,
            ).run(
                Path(job.source_path),
                store,
                metadata_overrides=metadata_overrides,
                dry_run=job.dry_run,
            )
            vector_sync = None
            if job.sync_vectors and not job.dry_run:
                chunks_by_id = {chunk.id: chunk for chunk in store.load()}
                chunks_to_upsert = [chunks_by_id[id] for id in report.chunks_upserted if id in chunks_by_id]
                sync_report = VectorIndexSync(
                    embedding_cache=self.embedding_cache,
                    embedding_ttl_seconds=self.embedding_ttl_seconds,
                ).sync(
                    create_vector_index(self.config.vector_index),
                    chunks_to_upsert=chunks_to_upsert,
                    chunk_ids_to_delete=list(report.chunks_deleted),
                )
                vector_sync = {
                    "vectors_upserted": sync_report.vectors_upserted,
                    "vectors_deleted": sync_report.vectors_deleted,
                }
            self.job_store.mark_succeeded(job_id, report=report, vector_sync=vector_sync)
            latency_ms = (time.perf_counter() - started_at) * 1000
            self.record_success(latency_ms)
            self.log_event(
                "ingest_job_completed",
                request_id=job.request_id,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                worker_id=self.worker_id,
                chunks_indexed=report.chunks_indexed,
                dry_run=job.dry_run,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            self.record_failure()
            self.job_store.mark_failed(job_id, error=str(exc))
            self.log_event(
                "ingest_job_failed",
                request_id=job.request_id,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                error=str(exc),
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
        finally:
            self.lease_store.release(lease_name, self.worker_id)

    def _tenant_metadata_filter(self, tenant_id: str | None) -> dict[str, str]:
        if tenant_id is None:
            return {}
        return {"tenant_id": tenant_id}

    def _metadata_overrides(self, job: IngestJobRecord) -> dict[str, str]:
        metadata = self._tenant_metadata_filter(job.tenant_id)
        if job.allowed_groups:
            metadata["allowed_groups"] = ",".join(job.allowed_groups)
        return metadata

    def _is_stale_running(self, job: IngestJobRecord) -> bool:
        if job.lease_expires_at is not None:
            return job.lease_expires_at <= self.now()
        return self.now() - job.updated_at >= self.config.jobs.running_timeout_seconds

    def _skip_job(self, job: IngestJobRecord, reason: str) -> None:
        self.record_skip(reason)
        self.log_event(
            "ingest_job_skipped",
            request_id=job.request_id,
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=job.status,
            attempt_count=job.attempt_count,
            reason=reason,
        )
