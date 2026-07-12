import json
from pathlib import Path

import pytest

from enterprise_rag import cli
from enterprise_rag.config import AppConfig
from enterprise_rag.jobs.ingest_jobs import JsonIngestJobStore
from enterprise_rag.models import Chunk
from enterprise_rag.observability.feedback import FeedbackRecord, JsonFeedbackStore
from enterprise_rag.storage.json_store import JsonChunkStore


class RecordingVectorSync:
    calls = []

    def __init__(self, embedding_model=None, embedding_cache=None, embedding_ttl_seconds=None) -> None:
        self.embedding_model = embedding_model
        self.embedding_cache = embedding_cache
        self.embedding_ttl_seconds = embedding_ttl_seconds

    def sync(self, vector_index: object, chunks_to_upsert: list[object], chunk_ids_to_delete: list[str]) -> object:
        self.calls.append((vector_index, chunks_to_upsert, chunk_ids_to_delete))
        return type(
            "SyncReport",
            (),
            {"vectors_upserted": len(chunks_to_upsert), "vectors_deleted": len(chunk_ids_to_delete)},
        )()


def test_ingest_cli_can_sync_changed_chunks_to_vector_index(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    index_path = tmp_path / "chunks.json"
    fake_vector_index = object()
    RecordingVectorSync.calls.clear()
    monkeypatch.setattr(cli, "create_vector_index", lambda config: fake_vector_index)
    monkeypatch.setattr(cli, "VectorIndexSync", RecordingVectorSync)

    cli.ingest(raw_dir, index_path, sync_vectors=True, allowed_groups=("security",))

    output = capsys.readouterr().out
    chunks = JsonChunkStore(index_path).load()
    assert "Vector sync report: upserted=1, deleted=0" in output
    assert chunks[0].metadata["allowed_groups"] == "security"
    assert RecordingVectorSync.calls[0][0] is fake_vector_index
    assert len(RecordingVectorSync.calls[0][1]) == 1
    assert RecordingVectorSync.calls[0][2] == []


def test_ingest_cli_dry_run_does_not_write_index(tmp_path: Path, capsys: object) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    index_path = tmp_path / "chunks.json"

    cli.ingest(raw_dir, index_path, dry_run=True)

    output = capsys.readouterr().out
    assert "Dry run: index was not written." in output
    assert not index_path.exists()


def test_run_job_cli_executes_persisted_ingest_job(tmp_path: Path, capsys: object) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    jobs_path = tmp_path / "jobs" / "ingest_jobs.json"
    index_path = tmp_path / "chunks.json"
    job_store = JsonIngestJobStore(jobs_path)
    job = job_store.create(str(raw_dir), tenant_id="acme", sync_vectors=False, request_id="req_123")

    cli.run_job(job.job_id, jobs_path, index_path)

    output = capsys.readouterr().out
    finished = JsonIngestJobStore(jobs_path).get(job.job_id)
    chunks = JsonChunkStore(index_path).load()
    assert finished is not None
    assert finished.status == "succeeded"
    assert chunks[0].metadata["tenant_id"] == "acme"
    assert f"Job {job.job_id} succeeded" in output
    assert "Ingest report: new=1" in output


def test_run_job_cli_exits_when_job_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No ingest job found"):
        cli.run_job("job_missing", tmp_path / "jobs.json", tmp_path / "chunks.json")


def test_worker_cli_runs_one_polling_pass(tmp_path: Path, capsys: object) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    jobs_path = tmp_path / "jobs" / "ingest_jobs.json"
    index_path = tmp_path / "chunks.json"
    job_store = JsonIngestJobStore(jobs_path)
    job = job_store.create(str(raw_dir), tenant_id=None, sync_vectors=False, request_id="req_123")

    cli.worker(jobs_path, index_path, once=True)

    output = capsys.readouterr().out
    finished = JsonIngestJobStore(jobs_path).get(job.job_id)
    assert finished is not None
    assert finished.status == "succeeded"
    assert "Worker scanned 1 jobs" in output


def test_eval_markdown_report_cli_writes_report_file(tmp_path: Path, capsys: object) -> None:
    index_path = tmp_path / "chunks.json"
    eval_path = tmp_path / "retrieval_eval.json"
    output_path = tmp_path / "reports" / "eval.md"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    eval_path.write_text(
        """
        [
          {
            "id": "hybrid",
            "query": "hybrid retrieval",
            "expected_chunk_ids": ["hybrid"]
          }
        ]
        """,
        encoding="utf-8",
    )

    cli.eval_markdown_report(
        eval_path,
        index_path,
        output_path,
        "Portfolio Eval",
        AppConfig(),
        k=1,
    )

    output = capsys.readouterr().out
    markdown = output_path.read_text(encoding="utf-8")
    assert f"Wrote evaluation report to {output_path}" in output
    assert "# Portfolio Eval" in markdown
    assert "- Recall@1: 1.00" in markdown
    assert "- Failures: 0" in markdown


def test_generate_eval_from_feedback_cli_writes_draft_cases(tmp_path: Path, capsys: object) -> None:
    feedback_path = tmp_path / "feedback.jsonl"
    output_path = tmp_path / "generated_from_feedback.json"
    JsonFeedbackStore(feedback_path).append(
        FeedbackRecord(
            feedback_id="fb_123",
            request_id="req_query_123",
            query="wrong citation query",
            answer="bad answer",
            rating="negative",
            labels=("wrong_citation",),
        )
    )

    cli.generate_eval_from_feedback(feedback_path, output_path, limit=10)

    output = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert f"Wrote 1 feedback draft eval cases to {output_path}" in output
    assert payload[0]["id"] == "feedback_1_wrong_citation_query"
    assert payload[0]["query"] == "wrong citation query"
