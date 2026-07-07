from enterprise_rag.config import VectorIndexConfig
from enterprise_rag.vector_index.factory import create_vector_index
from enterprise_rag.vector_index.in_memory import InMemoryVectorIndex
from enterprise_rag.vector_index.qdrant import QdrantVectorIndex


class FakeQdrantClient:
    def upsert(self, collection_name: str, points: list[object]) -> None:
        pass

    def query_points(self, collection_name: str, query: list[float], limit: int) -> list[object]:
        return []


def test_create_vector_index_defaults_to_memory() -> None:
    index = create_vector_index(VectorIndexConfig())

    assert isinstance(index, InMemoryVectorIndex)


def test_create_vector_index_builds_qdrant_with_injected_client() -> None:
    client = FakeQdrantClient()
    index = create_vector_index(
        VectorIndexConfig(
            provider="qdrant",
            collection_name="chunks",
            url="http://qdrant:6333",
        ),
        qdrant_client=client,
    )

    assert isinstance(index, QdrantVectorIndex)
    assert index.collection_name == "chunks"
    assert index.client is client


def test_create_vector_index_rejects_unknown_provider() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported vector index provider"):
        create_vector_index(VectorIndexConfig(provider="unknown"))
