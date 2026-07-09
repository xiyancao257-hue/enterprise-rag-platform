import json

from enterprise_rag.observability.audit import AuditEvent, JsonAuditLogger, NullAuditLogger


def test_json_audit_logger_writes_jsonl_events(tmp_path) -> None:
    path = tmp_path / "audit" / "audit.jsonl"
    logger = JsonAuditLogger(path)

    logger.log(
        AuditEvent(
            event_type="query.completed",
            request_id="req_123",
            tenant_id="acme",
            principal="api_key:hash",
            timestamp=100.0,
            attributes={"citation_chunk_ids": ["chunk1"]},
        )
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event_type"] == "query.completed"
    assert payload["request_id"] == "req_123"
    assert payload["tenant_id"] == "acme"
    assert payload["principal"] == "api_key:hash"
    assert payload["attributes"]["citation_chunk_ids"] == ["chunk1"]


def test_null_audit_logger_discards_events() -> None:
    NullAuditLogger().log(AuditEvent(event_type="query.completed", request_id="req_123"))
