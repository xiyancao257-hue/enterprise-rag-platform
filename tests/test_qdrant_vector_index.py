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
        self.collections: set[str] = set()
        self.created_collections = []
        self.upserts = []
        self.query_calls = []
        self.response = FakeQueryResponse(points=[FakePoint(id="chunk1", score=0.9)])

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections.add(collection_name)
        self.created_collections.append((collection_name, vectors_config))

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upserts.append((collection_name, points))

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        query_filter: object | None = None,
    ) -> FakeQueryResponse:
        self.query_calls.append((collection_name, query, limit, query_filter))
        return self.response


def test_qdrant_vector_index_adds_points() -> None:
    client = FakeQdrantClient()
    client.collections.add("chunks")
    index = QdrantVectorIndex(
        collection_name="chunks",
        client=client,
        point_factory=lambda id, vector, metadata: {"id": id, "vector": vector, "payload": metadata},
    )

    index.add("chunk1", [1.0, 0.0], metadata={"tenant_id": "acme"})

    assert client.upserts[0][0] == "chunks"
    assert client.upserts[0][1] == [{"id": "chunk1", "vector": [1.0, 0.0], "payload": {"tenant_id": "acme"}}]


def test_qdrant_vector_index_creates_missing_collection_from_vector_size() -> None:
    client = FakeQdrantClient()
    index = QdrantVectorIndex(
        collection_name="chunks",
        client=client,
        point_factory=lambda id, vector, metadata: {"id": id, "vector": vector, "payload": metadata},
        vector_params_factory=lambda vector_size: {"size": vector_size, "distance": "cosine"},
    )

    index.add("chunk1", [1.0, 0.0, 0.5])
    index.add("chunk2", [0.0, 1.0, 0.5])

    assert client.created_collections == [("chunks", {"size": 3, "distance": "cosine"})]
    assert [call[0] for call in client.upserts] == ["chunks", "chunks"]


def test_qdrant_vector_index_search_maps_results() -> None:
    client = FakeQdrantClient()
    index = QdrantVectorIndex(collection_name="chunks", client=client)

    results = index.search([1.0, 0.0], top_k=3)

    assert client.query_calls == [("chunks", [1.0, 0.0], 3, None)]
    assert results[0].id == "chunk1"
    assert results[0].score == 0.9
    assert results[0].rank == 1


def test_qdrant_vector_index_pushes_metadata_filters_to_query() -> None:
    client = FakeQdrantClient()
    index = QdrantVectorIndex(
        collection_name="chunks",
        client=client,
        filter_factory=lambda filters: {"must": filters},
    )

    index.search([1.0, 0.0], top_k=3, metadata_filters={"tenant_id": "acme"})

    assert client.query_calls == [("chunks", [1.0, 0.0], 3, {"must": {"tenant_id": "acme"}})]


def test_qdrant_vector_index_requires_optional_dependency_without_client() -> None:
    with pytest.raises(RuntimeError, match="enterprise-rag\\[qdrant\\]"):
        QdrantVectorIndex(collection_name="chunks")
