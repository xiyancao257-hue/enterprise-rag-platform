from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from enterprise_rag.config import AppConfig
from enterprise_rag.indexing.vector_sync import VectorIndexSync
from enterprise_rag.ingestion.pipeline import IncrementalIngestPipeline
from enterprise_rag.jobs.ingest_jobs import IngestJobStore
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
        log_event: Callable[..., None] | None = None,
    ) -> None:
        self.job_store = job_store
        self.index_path = index_path
        self.config = config
        self.record_failure = record_failure or (lambda: None)
        self.log_event = log_event or (lambda event, **fields: None)

    def run(self, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if job is None:
            return
        if job.status not in self.RUNNABLE_STATUSES:
            self.log_event(
                "ingest_job_skipped",
                request_id=job.request_id,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                status=job.status,
            )
            return

        self.job_store.mark_running(job_id)
        started_at = time.perf_counter()
        try:
            metadata_overrides = self._tenant_metadata_filter(job.tenant_id)
            store = JsonChunkStore(self.index_path)
            report = IncrementalIngestPipeline().run(
                Path(job.source_path),
                store,
                metadata_overrides=metadata_overrides,
            )
            vector_sync = None
            if job.sync_vectors:
                chunks_by_id = {chunk.id: chunk for chunk in store.load()}
                chunks_to_upsert = [chunks_by_id[id] for id in report.chunks_upserted if id in chunks_by_id]
                sync_report = VectorIndexSync().sync(
                    create_vector_index(self.config.vector_index),
                    chunks_to_upsert=chunks_to_upsert,
                    chunk_ids_to_delete=list(report.chunks_deleted),
                )
                vector_sync = {
                    "vectors_upserted": sync_report.vectors_upserted,
                    "vectors_deleted": sync_report.vectors_deleted,
                }
            self.job_store.mark_succeeded(job_id, report=report, vector_sync=vector_sync)
            self.log_event(
                "ingest_job_completed",
                request_id=job.request_id,
                job_id=job.job_id,
                tenant_id=job.tenant_id,
                chunks_indexed=report.chunks_indexed,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
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

    def _tenant_metadata_filter(self, tenant_id: str | None) -> dict[str, str]:
        if tenant_id is None:
            return {}
        return {"tenant_id": tenant_id}
