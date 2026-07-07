from dataclasses import dataclass

import pytest

from enterprise_rag.vector_index.qdrant import QdrantVectorIndex


@dataclass(frozen=True)
class FakePoint:
    id: str
    score: float


@dataclass(frozen=True)
class FakeQueryResponse:
    points: list[FakePoint]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upserts = []
        self.query_calls = []
        self.response = FakeQueryResponse(points=[FakePoint(id="chunk1", score=0.9)])

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upserts.append((collection_name, points))

    def query_points(self, collection_name: str, query: list[float], limit: int) -> FakeQueryResponse:
        self.query_calls.append((collection_name, query, limit))
        return self.response


def test_qdrant_vector_index_adds_points() -> None:
    client = FakeQdrantClient()
    index = QdrantVectorIndex(
        collection_name="chunks",
        client=client,
        point_factory=lambda id, vector: {"id": id, "vector": vector},
    )

    index.add("chunk1", [1.0, 0.0])

    assert client.upserts[0][0] == "chunks"
    assert client.upserts[0][1] == [{"id": "chunk1", "vector": [1.0, 0.0]}]


def test_qdrant_vector_index_search_maps_results() -> None:
    client = FakeQdrantClient()
    index = QdrantVectorIndex(collection_name="chunks", client=client)

    results = index.search([1.0, 0.0], top_k=3)

    assert client.query_calls == [("chunks", [1.0, 0.0], 3)]
    assert results[0].id == "chunk1"
    assert results[0].score == 0.9
    assert results[0].rank == 1


def test_qdrant_vector_index_requires_optional_dependency_without_client() -> None:
    with pytest.raises(RuntimeError, match="enterprise-rag\\[qdrant\\]"):
        QdrantVectorIndex(collection_name="chunks")
