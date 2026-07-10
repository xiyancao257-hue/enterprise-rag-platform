from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from enterprise_rag.ingestion.loaders import FilteredDocument
from enterprise_rag.ingestion.pipeline import IngestReport


@dataclass
class IngestJobRecord:
    job_id: str
    status: str
    source_path: str
    tenant_id: str | None
    allowed_groups: tuple[str, ...]
    sync_vectors: bool
    dry_run: bool
    request_id: str
    created_at: float
    updated_at: float
    attempt_count: int = 0
    max_attempts: int = 3
    report: IngestReport | None = None
    vector_sync: dict[str, int] | None = None
    error: str | None = None


class IngestJobStore(Protocol):
    def create(
        self,
        source_path: str,
        tenant_id: str | None,
        sync_vectors: bool,
        request_id: str,
        allowed_groups: tuple[str, ...] = (),
        dry_run: bool = False,
    ) -> IngestJobRecord:
        """Create a queued ingest job."""

    def get(self, job_id: str) -> IngestJobRecord | None:
        """Read one ingest job."""

    def list(self) -> list[IngestJobRecord]:
        """List ingest jobs."""

    def mark_running(self, job_id: str) -> None:
        """Mark an ingest job as running."""

    def mark_succeeded(
        self,
        job_id: str,
        report: IngestReport,
        vector_sync: dict[str, int] | None = None,
    ) -> None:
        """Mark an ingest job as succeeded."""

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark an ingest job as failed."""

    def mark_canceled(self, job_id: str) -> None:
        """Mark an ingest job as canceled."""


class InMemoryIngestJobStore:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self.now = now or time.time
        self.jobs: dict[str, IngestJobRecord] = {}

    def create(
        self,
        source_path: str,
        tenant_id: str | None,
        sync_vectors: bool,
        request_id: str,
        allowed_groups: tuple[str, ...] = (),
        dry_run: bool = False,
    ) -> IngestJobRecord:
        job = _new_job(
            source_path,
            tenant_id,
            sync_vectors,
            request_id,
            self.now(),
            allowed_groups=allowed_groups,
            dry_run=dry_run,
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> IngestJobRecord | None:
        return self.jobs.get(job_id)

    def list(self) -> list[IngestJobRecord]:
        return list(self.jobs.values())

    def mark_running(self, job_id: str) -> None:
        job = self.jobs[job_id]
        self._update(job_id, status="running", attempt_count=job.attempt_count + 1)

    def mark_succeeded(
        self,
        job_id: str,
        report: IngestReport,
        vector_sync: dict[str, int] | None = None,
    ) -> None:
        self._update(job_id, status="succeeded", report=report, vector_sync=vector_sync, error=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", error=error)

    def mark_canceled(self, job_id: str) -> None:
        self._update(job_id, status="canceled", error=None)

    def _update(self, job_id: str, **changes: object) -> None:
        job = self.jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = self.now()


class JsonIngestJobStore:
    def __init__(self, path: Path, now: Callable[[], float] | None = None) -> None:
        self.path = path
        self.now = now or time.time

    def create(
        self,
        source_path: str,
        tenant_id: str | None,
        sync_vectors: bool,
        request_id: str,
        allowed_groups: tuple[str, ...] = (),
        dry_run: bool = False,
    ) -> IngestJobRecord:
        jobs = self._load()
        job = _new_job(
            source_path,
            tenant_id,
            sync_vectors,
            request_id,
            self.now(),
            allowed_groups=allowed_groups,
            dry_run=dry_run,
        )
        jobs[job.job_id] = job
        self._save(jobs)
        return job

    def get(self, job_id: str) -> IngestJobRecord | None:
        return self._load().get(job_id)

    def list(self) -> list[IngestJobRecord]:
        return list(self._load().values())

    def mark_running(self, job_id: str) -> None:
        jobs = self._load()
        job = jobs[job_id]
        self._update(job_id, status="running", attempt_count=job.attempt_count + 1)

    def mark_succeeded(
        self,
        job_id: str,
        report: IngestReport,
        vector_sync: dict[str, int] | None = None,
    ) -> None:
        self._update(job_id, status="succeeded", report=report, vector_sync=vector_sync, error=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", error=error)

    def mark_canceled(self, job_id: str) -> None:
        self._update(job_id, status="canceled", error=None)

    def _update(self, job_id: str, **changes: object) -> None:
        jobs = self._load()
        job = jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = self.now()
        jobs[job_id] = job
        self._save(jobs)

    def _load(self) -> dict[str, IngestJobRecord]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {job_id: _job_from_dict(item) for job_id, item in payload.items()}

    def _save(self, jobs: dict[str, IngestJobRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {job_id: _job_to_dict(job) for job_id, job in jobs.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _new_job(
    source_path: str,
    tenant_id: str | None,
    sync_vectors: bool,
    request_id: str,
    now: float,
    allowed_groups: tuple[str, ...] = (),
    dry_run: bool = False,
) -> IngestJobRecord:
    return IngestJobRecord(
        job_id=f"job_{uuid4().hex}",
        status="queued",
        source_path=source_path,
        tenant_id=tenant_id,
        allowed_groups=allowed_groups,
        sync_vectors=sync_vectors,
        dry_run=dry_run,
        request_id=request_id,
        created_at=now,
        updated_at=now,
        attempt_count=0,
        max_attempts=3,
    )


def _job_to_dict(job: IngestJobRecord) -> dict[str, object]:
    payload = asdict(job)
    if job.report is not None:
        payload["report"] = asdict(job.report)
    return payload


def _job_from_dict(item: dict[str, object]) -> IngestJobRecord:
    report_data = item.get("report")
    report = _report_from_dict(report_data) if isinstance(report_data, dict) else None
    vector_sync = item.get("vector_sync")
    parsed_vector_sync = (
        {str(key): int(value) for key, value in vector_sync.items()} if isinstance(vector_sync, dict) else None
    )
    return IngestJobRecord(
        job_id=str(item["job_id"]),
        status=str(item["status"]),
        source_path=str(item["source_path"]),
        tenant_id=str(item["tenant_id"]) if item.get("tenant_id") is not None else None,
        allowed_groups=tuple(str(group) for group in item.get("allowed_groups", ())),
        sync_vectors=bool(item["sync_vectors"]),
        dry_run=bool(item.get("dry_run", False)),
        request_id=str(item["request_id"]),
        created_at=float(item["created_at"]),
        updated_at=float(item["updated_at"]),
        attempt_count=int(item.get("attempt_count", 0)),
        max_attempts=int(item.get("max_attempts", 3)),
        report=report,
        vector_sync=parsed_vector_sync,
        error=str(item["error"]) if item.get("error") is not None else None,
    )


def _report_from_dict(item: dict[str, object]) -> IngestReport:
    return IngestReport(
        documents_loaded=int(item["documents_loaded"]),
        documents_new=int(item["documents_new"]),
        documents_updated=int(item["documents_updated"]),
        documents_unchanged=int(item["documents_unchanged"]),
        documents_deleted=int(item["documents_deleted"]),
        documents_filtered=int(item["documents_filtered"]),
        chunks_indexed=int(item["chunks_indexed"]),
        chunks_upserted=tuple(str(value) for value in item.get("chunks_upserted", ())),
        chunks_deleted=tuple(str(value) for value in item.get("chunks_deleted", ())),
        filter_reasons={str(key): int(value) for key, value in item.get("filter_reasons", {}).items()},
        filtered_documents=tuple(
            FilteredDocument(source_path=str(value["source_path"]), reason=str(value["reason"]))
            for value in item.get("filtered_documents", ())
            if isinstance(value, dict)
        ),
        dry_run=bool(item.get("dry_run", False)),
    )
