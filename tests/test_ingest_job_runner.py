from pathlib import Path

from enterprise_rag.config import AppConfig, JobsConfig
from enterprise_rag.ingestion.pipeline import IngestReport
from enterprise_rag.jobs.ingest_jobs import InMemoryIngestJobStore
from enterprise_rag.jobs.runner import IngestJobRunner
from enterprise_rag.storage.json_store import JsonChunkStore


def test_ingest_job_runner_marks_job_succeeded_and_indexes_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    index_path = tmp_path / "chunks.json"
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id="acme", sync_vectors=False, request_id="req_123")
    events = []

    IngestJobRunner(
        job_store=store,
        index_path=index_path,
        config=AppConfig(),
        log_event=lambda event, **fields: events.append((event, fields)),
    ).run(job.job_id)

    finished = store.get(job.job_id)
    chunks = JsonChunkStore(index_path).load()
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.report is not None
    assert finished.report.documents_new == 1
    assert chunks[0].metadata["tenant_id"] == "acme"
    assert events[0][0] == "ingest_job_completed"


def test_ingest_job_runner_marks_job_failed_and_records_failure(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    failures = []
    events = []

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path,
        config=AppConfig(),
        record_failure=lambda: failures.append("failed"),
        log_event=lambda event, **fields: events.append((event, fields)),
    ).run(job.job_id)

    failed = store.get(job.job_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.attempt_count == 1
    assert failed.error is not None
    assert failures == ["failed"]
    assert events[0][0] == "ingest_job_failed"


def test_ingest_job_runner_skips_succeeded_job_duplicate_delivery(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    store.mark_succeeded(
        job.job_id,
        report=IngestReport(
            documents_loaded=1,
            documents_new=1,
            documents_updated=0,
            documents_unchanged=0,
            documents_deleted=0,
            documents_filtered=0,
            chunks_indexed=1,
        ),
    )
    events = []

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path / "chunks.json",
        config=AppConfig(),
        log_event=lambda event, **fields: events.append((event, fields)),
    ).run(job.job_id)

    skipped = store.get(job.job_id)
    assert skipped is not None
    assert skipped.status == "succeeded"
    assert not (tmp_path / "chunks.json").exists()
    assert events == [
        (
            "ingest_job_skipped",
            {
                "request_id": "req_123",
                "job_id": job.job_id,
                "tenant_id": None,
                "status": "succeeded",
                "attempt_count": 0,
                "reason": "succeeded",
            },
        )
    ]


def test_ingest_job_runner_skips_running_job_duplicate_delivery(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    store.mark_running(job.job_id)
    events = []

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path / "chunks.json",
        config=AppConfig(),
        log_event=lambda event, **fields: events.append((event, fields)),
    ).run(job.job_id)

    skipped = store.get(job.job_id)
    assert skipped is not None
    assert skipped.status == "running"
    assert events[0][0] == "ingest_job_skipped"
    assert events[0][1]["status"] == "running"
    assert events[0][1]["attempt_count"] == 1


def test_ingest_job_runner_recovers_stale_running_job(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore(now=lambda: 100.0)
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    store.mark_running(job.job_id)

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path / "chunks.json",
        config=AppConfig(jobs=JobsConfig(running_timeout_seconds=60)),
        now=lambda: 200.0,
    ).run(job.job_id)

    recovered = store.get(job.job_id)
    assert recovered is not None
    assert recovered.status == "succeeded"
    assert recovered.attempt_count == 2


def test_ingest_job_runner_allows_failed_job_retry(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    IngestJobRunner(
        job_store=store,
        index_path=tmp_path,
        config=AppConfig(),
    ).run(job.job_id)

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path / "chunks.json",
        config=AppConfig(),
    ).run(job.job_id)

    retried = store.get(job.job_id)
    assert retried is not None
    assert retried.status == "succeeded"
    assert retried.attempt_count == 2
    assert retried.report is not None
    assert retried.report.documents_new == 1


def test_ingest_job_runner_skips_failed_job_after_max_attempts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = InMemoryIngestJobStore()
    job = store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")
    stored = store.get(job.job_id)
    assert stored is not None
    stored.attempt_count = stored.max_attempts
    store.mark_failed(job.job_id, error="permanent failure")
    events = []

    IngestJobRunner(
        job_store=store,
        index_path=tmp_path / "chunks.json",
        config=AppConfig(),
        log_event=lambda event, **fields: events.append((event, fields)),
    ).run(job.job_id)

    exhausted = store.get(job.job_id)
    assert exhausted is not None
    assert exhausted.status == "failed"
    assert exhausted.attempt_count == exhausted.max_attempts
    assert not (tmp_path / "chunks.json").exists()
    assert events[0][0] == "ingest_job_retry_exhausted"
    assert events[0][1]["attempt_count"] == exhausted.max_attempts
