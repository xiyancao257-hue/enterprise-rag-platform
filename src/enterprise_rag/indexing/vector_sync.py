from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.models import Chunk
from enterprise_rag.vector_index.base import VectorIndex


@dataclass(frozen=True)
class VectorSyncReport:
    vectors_upserted: int
    vectors_deleted: int


class VectorIndexSync:
    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or HashingEmbeddingModel()

    def sync(
        self,
        vector_index: VectorIndex,
        chunks_to_upsert: list[Chunk],
        chunk_ids_to_delete: list[str],
    ) -> VectorSyncReport:
        vector_index.delete(chunk_ids_to_delete)
        for chunk in chunks_to_upsert:
            vector_index.add(chunk.id, self.embedding_model.embed(chunk.text), metadata=chunk.metadata)
        return VectorSyncReport(
            vectors_upserted=len(chunks_to_upsert),
            vectors_deleted=len(chunk_ids_to_delete),
        )
