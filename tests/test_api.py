from fastapi.testclient import TestClient

from enterprise_rag.api.app import create_app
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
        "status": "ok",
        "chunk_count": 1,
        "vector_index_provider": "memory",
    }


def test_query_returns_answer_plan_and_citations(tmp_path) -> None:
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

    response = client.post(
        "/query",
        json={
            "query": "hybrid retrival",
            "top_k": 1,
            "include_trace": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert "BM25" in payload["answer"]
    assert payload["query_plan"]["corrections"] == {"retrival": "retrieval"}
    assert payload["citations"][0]["chunk_id"] == "hybrid"
    assert payload["trace"]["retrieved"]


def test_query_returns_404_when_index_is_empty(tmp_path) -> None:
    client = TestClient(create_app(index_path=tmp_path / "missing.json"))

    response = client.post("/query", json={"query": "hybrid retrieval"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No chunks found. Run ingestion before querying."
