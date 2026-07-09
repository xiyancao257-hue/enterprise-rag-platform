from __future__ import annotations

from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.vector_index.base import VectorIndex
from enterprise_rag.vector_index.in_memory import InMemoryVectorIndex


class HashingVectorRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model: EmbeddingModel | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self.chunks = chunks
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.vector_index = vector_index or InMemoryVectorIndex()
        for chunk in chunks:
            self.vector_index.add(chunk.id, self.embedding_model.embed(chunk.text), metadata=chunk.metadata)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        query_vector = self.embedding_model.embed(query)
        results = self.vector_index.search(query_vector, top_k=top_k, metadata_filters=metadata_filters)
        return [
            SearchHit(
                chunk=self.chunks_by_id[result.id],
                score=result.score,
                retriever="vector",
                rank=result.rank,
            )
            for result in results
            if result.id in self.chunks_by_id
        ]
