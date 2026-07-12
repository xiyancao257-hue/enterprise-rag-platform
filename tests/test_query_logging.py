import json

from enterprise_rag.models import Chunk
from enterprise_rag.observability.query_logging import (
    QueryLogger,
    build_query_log_record,
)
from enterprise_rag.rag.pipeline import RagPipeline


def test_build_query_log_record_summarizes_trace() -> None:
    chunks = [
        Chunk(
            id="auth",
            document_id="doc1",
            text="Auth Service uses Rate Limit Policy.",
        )
    ]
    answer, trace = RagPipeline(chunks).answer_for_user_with_trace(
        "Auth Service",
        top_k=1,
        user_groups={"engineering"},
    )

    record = build_query_log_record(
        answer,
        trace,
        top_k=1,
        enable_graph=False,
        graph_max_hops=2,
        user_groups={"engineering"},
        index_version="idx-test",
    )

    assert record.query == "Auth Service"
    assert record.normalized_query == "Auth Service"
    assert record.top_k == 1
    assert record.enable_graph is False
    assert record.index_version == "idx-test"
    assert record.user_groups == ("engineering",)
    assert record.retrieved_chunk_ids == ("auth",)
    assert record.final_chunk_ids == ("auth",)
    assert record.insufficient_evidence is False


def test_query_logger_writes_jsonl(tmp_path) -> None:
    chunks = [
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
        )
    ]
    answer, trace = RagPipeline(chunks).answer_for_user_with_trace("AUTH-429", top_k=1)
    record = build_query_log_record(
        answer,
        trace,
        top_k=1,
        enable_graph=True,
        graph_max_hops=3,
        index_version="idx-jsonl",
    )
    log_path = tmp_path / "logs" / "query_log.jsonl"

    QueryLogger(log_path).log(record)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["query"] == "AUTH-429"
    assert data["retrieved_chunk_ids"] == ["policy"]
    assert data["final_chunk_ids"] == ["policy"]
    assert data["enable_graph"] is True
    assert data["graph_max_hops"] == 3
    assert data["index_version"] == "idx-jsonl"
