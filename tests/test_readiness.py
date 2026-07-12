import json

from enterprise_rag.config import (
    ApiSecurityConfig,
    AppConfig,
    AuditConfig,
    CacheConfig,
    EmbeddingConfig,
    IngestionConfig,
    LeaseConfig,
    LLMConfig,
    VectorIndexConfig,
)
from enterprise_rag.evaluation.readiness import build_readiness_report, format_readiness_report
from enterprise_rag.models import Chunk
from enterprise_rag.storage.json_store import JsonChunkStore


def test_build_readiness_report_with_eval_log_and_self_healing_artifacts(tmp_path) -> None:
    index_path = tmp_path / "chunks.json"
    eval_path = tmp_path / "retrieval_eval.json"
    query_log_path = tmp_path / "query_log.jsonl"
    self_healing_dir = tmp_path / "self_healing"
    self_healing_dir.mkdir()
    chunks = [
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
        )
    ]
    JsonChunkStore(index_path).save(chunks)
    eval_path.write_text(
        json.dumps(
            [
                {
                    "id": "auth_429",
                    "query": "What is AUTH-429?",
                    "expected_text_contains": ["Rate Limit Policy defines AUTH-429."],
                }
            ]
        ),
        encoding="utf-8",
    )
    query_log_path.write_text(
        json.dumps(
            {
                "query": "What is AUTH-429?",
                "retrieved_chunk_ids": ["policy"],
                "final_chunk_ids": ["policy"],
                "insufficient_evidence": False,
                "enable_graph": False,
            }
        ),
        encoding="utf-8",
    )
    (self_healing_dir / "generated_from_logs.json").write_text("[]", encoding="utf-8")
    (self_healing_dir / "generated_with_suggestions.json").write_text("[]", encoding="utf-8")

    report = build_readiness_report(
        chunks,
        index_path=index_path,
        eval_path=eval_path,
        query_log_path=query_log_path,
        self_healing_dir=self_healing_dir,
        config=AppConfig(
            api_security=ApiSecurityConfig(require_api_key=True),
            audit=AuditConfig(enabled=True),
            vector_index=VectorIndexConfig(provider="qdrant"),
            cache=CacheConfig(provider="redis"),
            leases=LeaseConfig(provider="redis"),
            llm=LLMConfig(max_retries=2, circuit_breaker_failure_threshold=3),
            embedding=EmbeddingConfig(max_retries=1),
            ingestion=IngestionConfig(allowed_source_roots=("data/raw",)),
        ),
        k=1,
    )

    assert report.index_present is True
    assert report.chunk_count == 1
    assert report.eval_present is True
    assert report.eval_case_count == 1
    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 1.0
    assert report.mrr == 1.0
    assert report.query_log_present is True
    assert report.log_analysis is not None
    assert report.log_analysis.total_queries == 1
    assert report.self_healing_draft_present is True
    assert report.self_healing_suggestions_present is True
    assert {check.name: check.status for check in report.enterprise_checks} == {
        "index": "pass",
        "api_auth": "pass",
        "audit_logging": "pass",
        "vector_index": "pass",
        "cache": "pass",
        "leases": "pass",
        "provider_resilience": "pass",
        "eval_coverage": "pass",
        "query_logging": "pass",
        "self_healing": "pass",
        "ingestion_policy": "pass",
    }


def test_format_readiness_report_for_missing_artifacts() -> None:
    from pathlib import Path

    report = build_readiness_report(
        [],
        index_path=Path("missing_index.json"),
        eval_path=None,
        query_log_path=None,
        self_healing_dir=None,
        k=5,
    )

    formatted = format_readiness_report(report, k=5)

    assert "Readiness Report" in formatted
    assert "- index: missing" in formatted
    assert "- chunks: 0" in formatted
    assert "- eval file: missing" in formatted
    assert "- query log: missing" in formatted
    assert "Enterprise Checks" in formatted
    assert "- index: fail - no chunks indexed" in formatted
    assert "- api_auth: warn - API key is not required" in formatted
    assert "Run ingestion before demo or deployment." in formatted
    assert "Resolve failing enterprise readiness checks before production rollout." in formatted
    assert "Run pytest before demo or deployment." in formatted
