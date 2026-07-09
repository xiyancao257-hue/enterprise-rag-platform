import json
import logging
from hashlib import sha256

from fastapi.testclient import TestClient

from enterprise_rag.api.app import API_KEY_HEADER, REQUEST_ID_HEADER, create_app
from enterprise_rag.config import ApiSecurityConfig, AppConfig
from enterprise_rag.models import Chunk
from enterprise_rag.storage.json_store import JsonChunkStore


def test_health_reports_index_status(tmp_path) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="chunk1",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    client = TestClient(create_app(index_path=index_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "request_id": response.headers[REQUEST_ID_HEADER],
        "status": "ok",
        "chunk_count": 1,
        "vector_index_provider": "memory",
    }


def test_query_returns_answer_plan_citations_and_request_id(tmp_path, caplog) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            ),
            Chunk(
                id="cleaning",
                document_id="doc1",
                text="Dirty data cleaning removes repeated headers and OCR noise.",
            ),
        ]
    )
    client = TestClient(create_app(index_path=index_path))
    caplog.set_level(logging.INFO, logger="enterprise_rag.api")

    response = client.post(
        "/query",
        headers={REQUEST_ID_HEADER: "req_test_123"},
        json={
            "query": "hybrid retrival",
            "top_k": 1,
            "include_trace": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "req_test_123"
    assert payload["request_id"] == "req_test_123"
    assert "BM25" in payload["answer"]
    assert payload["query_plan"]["corrections"] == {"retrival": "retrieval"}
    assert payload["citations"][0]["chunk_id"] == "hybrid"
    assert payload["trace"]["retrieved"]
    events = [json.loads(record.message) for record in caplog.records]
    query_completed = next(event for event in events if event["event"] == "query_completed")
    assert {
        "event": "query_completed",
        "request_id": "req_test_123",
        "top_k": 1,
        "citation_count": 1,
        "include_trace": True,
        "vector_index_provider": "memory",
    }.items() <= query_completed.items()


def test_query_returns_404_when_index_is_empty(tmp_path) -> None:
    client = TestClient(create_app(index_path=tmp_path / "missing.json"))

    response = client.post("/query", json={"query": "hybrid retrieval"})

    assert response.status_code == 404
    assert response.headers[REQUEST_ID_HEADER].startswith("req_")
    assert response.json()["detail"] == "No chunks found. Run ingestion before querying."


def test_metrics_reports_http_query_and_failure_counts(tmp_path) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    client = TestClient(create_app(index_path=index_path))

    initial_metrics = client.get("/metrics").text
    assert "enterprise_rag_http_requests_total 0" in initial_metrics
    assert "enterprise_rag_query_requests_total 0" in initial_metrics

    client.get("/health")
    client.post("/query", json={"query": "hybrid retrieval", "top_k": 1})

    metrics = client.get("/metrics").text
    assert "enterprise_rag_http_requests_total 3" in metrics
    assert "enterprise_rag_query_requests_total 1" in metrics
    assert "enterprise_rag_query_failures_total 0" in metrics
    assert "enterprise_rag_query_latency_ms_count 1" in metrics
    assert "enterprise_rag_query_citations_total 1" in metrics


def test_metrics_records_failed_query(tmp_path) -> None:
    client = TestClient(create_app(index_path=tmp_path / "missing.json"))

    client.post("/query", json={"query": "hybrid retrieval"})

    metrics = client.get("/metrics").text
    assert "enterprise_rag_query_requests_total 1" in metrics
    assert "enterprise_rag_query_failures_total 1" in metrics


def test_health_remains_public_when_api_key_is_required(tmp_path) -> None:
    app = create_app(
        index_path=tmp_path / "missing.json",
        config=AppConfig(api_security=ApiSecurityConfig(require_api_key=True)),
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200


def test_query_rejects_missing_or_wrong_api_key(tmp_path) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    app = create_app(
        index_path=index_path,
        config=AppConfig(
            api_security=ApiSecurityConfig(
                require_api_key=True,
                api_key_hashes=(_hash_test_key("correct-key"),),
            )
        ),
    )
    client = TestClient(app)

    missing_response = client.post("/query", json={"query": "hybrid retrieval"})
    wrong_response = client.post(
        "/query",
        headers={API_KEY_HEADER: "wrong-key"},
        json={"query": "hybrid retrieval"},
    )

    assert missing_response.status_code == 401
    assert wrong_response.status_code == 401
    assert missing_response.json()["detail"] == "Invalid or missing API key."


def test_query_accepts_hashed_api_key_or_bearer_token(tmp_path) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    app = create_app(
        index_path=index_path,
        config=AppConfig(
            api_security=ApiSecurityConfig(
                require_api_key=True,
                api_key_hashes=(_hash_test_key("correct-key"),),
            )
        ),
    )
    client = TestClient(app)

    header_response = client.post(
        "/query",
        headers={API_KEY_HEADER: "correct-key"},
        json={"query": "hybrid retrieval", "top_k": 1},
    )
    bearer_response = client.post(
        "/query",
        headers={"Authorization": "Bearer correct-key"},
        json={"query": "hybrid retrieval", "top_k": 1},
    )

    assert header_response.status_code == 200
    assert bearer_response.status_code == 200


def test_metrics_requires_api_key_when_auth_enabled(tmp_path) -> None:
    app = create_app(
        index_path=tmp_path / "missing.json",
        config=AppConfig(
            api_security=ApiSecurityConfig(
                require_api_key=True,
                api_key_hashes=(_hash_test_key("correct-key"),),
            )
        ),
    )
    client = TestClient(app)

    denied_response = client.get("/metrics")
    allowed_response = client.get("/metrics", headers={API_KEY_HEADER: "correct-key"})

    assert denied_response.status_code == 401
    assert allowed_response.status_code == 200
    assert "enterprise_rag_http_requests_total" in allowed_response.text


def test_query_accepts_api_key_from_environment(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "chunks.json"
    JsonChunkStore(index_path).save(
        [
            Chunk(
                id="hybrid",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
            )
        ]
    )
    monkeypatch.setenv("ENTERPRISE_RAG_TEST_KEYS", "env-key-1,env-key-2")
    app = create_app(
        index_path=index_path,
        config=AppConfig(
            api_security=ApiSecurityConfig(
                require_api_key=True,
                api_key_env_var="ENTERPRISE_RAG_TEST_KEYS",
            )
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/query",
        headers={API_KEY_HEADER: "env-key-2"},
        json={"query": "hybrid retrieval", "top_k": 1},
    )

    assert response.status_code == 200


def _hash_test_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()
