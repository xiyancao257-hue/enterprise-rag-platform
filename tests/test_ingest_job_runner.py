from pathlib import Path

from enterprise_rag.config import AppConfig
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
    assert failed.error is not None
    assert failures == ["failed"]
    assert events[0][0] == "ingest_job_failed"
