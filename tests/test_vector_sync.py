from enterprise_rag.indexing.vector_sync import VectorIndexSync
from enterprise_rag.models import Chunk


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text))]


class RecordingVectorIndex:
    def __init__(self) -> None:
        self.calls = []

    def add(self, id: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        self.calls.append(("add", id, vector, metadata))

    def delete(self, ids: list[str]) -> None:
        self.calls.append(("delete", ids))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[object]:
        return []


def test_vector_index_sync_deletes_stale_vectors_before_upserting_chunks() -> None:
    index = RecordingVectorIndex()
    chunk = Chunk(
        id="new_chunk",
        document_id="doc1",
        text="updated text",
        metadata={"tenant_id": "acme", "source_path": "guide.md"},
    )

    report = VectorIndexSync(embedding_model=FakeEmbeddingModel()).sync(
        index,
        chunks_to_upsert=[chunk],
        chunk_ids_to_delete=["old_chunk"],
    )

    assert report.vectors_deleted == 1
    assert report.vectors_upserted == 1
    assert index.calls == [
        ("delete", ["old_chunk"]),
        ("add", "new_chunk", [12.0], {"tenant_id": "acme", "source_path": "guide.md"}),
    ]


def test_vector_index_sync_can_use_cached_embedding_model_for_duplicate_text() -> None:
    from enterprise_rag.cache.in_memory import InMemoryCache
    from enterprise_rag.embeddings.cached import CachedEmbeddingModel

    index = RecordingVectorIndex()
    base_model = FakeEmbeddingModel()
    cache = InMemoryCache()
    embedding_model = CachedEmbeddingModel(base_model, cache=cache, model_id="fake")
    chunks = [
        Chunk(id="chunk1", document_id="doc1", text="same text"),
        Chunk(id="chunk2", document_id="doc2", text="same text"),
    ]

    report = VectorIndexSync(embedding_model=embedding_model).sync(
        index,
        chunks_to_upsert=chunks,
        chunk_ids_to_delete=[],
    )

    assert report.vectors_upserted == 2
    assert base_model.calls == 1
    assert cache.hits == 1
    assert cache.misses == 1
