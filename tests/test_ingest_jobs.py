from pathlib import Path

from enterprise_rag.ingestion.loaders import FilteredDocument
from enterprise_rag.ingestion.pipeline import IngestReport
from enterprise_rag.jobs.ingest_jobs import JsonIngestJobStore


def test_json_ingest_job_store_persists_created_jobs(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs" / "ingest_jobs.json"
    store = JsonIngestJobStore(store_path, now=lambda: 100.0)

    job = store.create(
        source_path="data/raw",
        tenant_id="acme",
        sync_vectors=True,
        request_id="req_123",
        allowed_groups=("legal", "security"),
    )
    reloaded = JsonIngestJobStore(store_path).get(job.job_id)

    assert reloaded is not None
    assert reloaded.status == "queued"
    assert reloaded.source_path == "data/raw"
    assert reloaded.tenant_id == "acme"
    assert reloaded.allowed_groups == ("legal", "security")
    assert reloaded.sync_vectors is True
    assert reloaded.dry_run is False
    assert reloaded.attempt_count == 0
    assert reloaded.max_attempts == 3
    assert reloaded.created_at == 100.0


def test_json_ingest_job_store_persists_dry_run_jobs(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs" / "ingest_jobs.json"
    store = JsonIngestJobStore(store_path)

    job = store.create("data/raw", tenant_id=None, sync_vectors=True, request_id="req_123", dry_run=True)
    reloaded = JsonIngestJobStore(store_path).get(job.job_id)

    assert reloaded is not None
    assert reloaded.dry_run is True
    assert reloaded.sync_vectors is True


def test_json_ingest_job_store_lists_jobs(tmp_path: Path) -> None:
    store = JsonIngestJobStore(tmp_path / "jobs" / "ingest_jobs.json")
    first = store.create("data/a", tenant_id="acme", sync_vectors=False, request_id="req_1")
    second = store.create("data/b", tenant_id="globex", sync_vectors=True, request_id="req_2")

    jobs = store.list()

    assert {job.job_id for job in jobs} == {first.job_id, second.job_id}


def test_json_ingest_job_store_persists_status_updates(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs" / "ingest_jobs.json"
    clock = iter([100.0, 101.0, 102.0])
    store = JsonIngestJobStore(store_path, now=lambda: next(clock))
    job = store.create("data/raw", tenant_id=None, sync_vectors=False, request_id="req_123")

    store.mark_running(job.job_id)
    store.mark_succeeded(
        job.job_id,
        report=IngestReport(
            documents_loaded=2,
            documents_new=1,
            documents_updated=1,
            documents_unchanged=0,
            documents_deleted=0,
            documents_filtered=0,
            chunks_indexed=5,
            chunks_upserted=("chunk_new",),
            chunks_deleted=("chunk_old",),
            filter_reasons={"unsupported_extension": 1},
            filtered_documents=(FilteredDocument(source_path="data/raw/image.png", reason="unsupported_extension"),),
        ),
        vector_sync={"vectors_upserted": 1, "vectors_deleted": 1},
    )

    reloaded = JsonIngestJobStore(store_path).get(job.job_id)
    assert reloaded is not None
    assert reloaded.status == "succeeded"
    assert reloaded.updated_at == 102.0
    assert reloaded.attempt_count == 1
    assert reloaded.report is not None
    assert reloaded.report.chunks_upserted == ("chunk_new",)
    assert reloaded.report.chunks_deleted == ("chunk_old",)
    assert reloaded.report.filter_reasons == {"unsupported_extension": 1}
    assert reloaded.report.filtered_documents == (
        FilteredDocument(source_path="data/raw/image.png", reason="unsupported_extension"),
    )
    assert reloaded.vector_sync == {"vectors_upserted": 1, "vectors_deleted": 1}


def test_json_ingest_job_store_persists_failures(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs" / "ingest_jobs.json"
    store = JsonIngestJobStore(store_path)
    job = store.create("data/raw", tenant_id=None, sync_vectors=False, request_id="req_123")

    store.mark_failed(job.job_id, error="OCR failed")

    reloaded = JsonIngestJobStore(store_path).get(job.job_id)
    assert reloaded is not None
    assert reloaded.status == "failed"
    assert reloaded.error == "OCR failed"


def test_json_ingest_job_store_preserves_attempt_metadata(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs" / "ingest_jobs.json"
    store = JsonIngestJobStore(store_path)
    job = store.create("data/raw", tenant_id=None, sync_vectors=False, request_id="req_123")

    store.mark_running(job.job_id)
    store.mark_failed(job.job_id, error="temporary failure")

    reloaded = JsonIngestJobStore(store_path).get(job.job_id)
    assert reloaded is not None
    assert reloaded.attempt_count == 1
    assert reloaded.max_attempts == 3
