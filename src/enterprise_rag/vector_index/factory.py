from __future__ import annotations

from enterprise_rag.config import VectorIndexConfig
from enterprise_rag.vector_index.base import VectorIndex
from enterprise_rag.vector_index.in_memory import InMemoryVectorIndex
from enterprise_rag.vector_index.qdrant import QdrantClientLike, QdrantVectorIndex


def create_vector_index(config: VectorIndexConfig, qdrant_client: QdrantClientLike | None = None) -> VectorIndex:
    provider = config.provider.lower()
    if provider == "memory":
        return InMemoryVectorIndex()
    if provider == "qdrant":
        return QdrantVectorIndex(
            collection_name=config.collection_name,
            client=qdrant_client,
            url=config.url,
        )
    raise ValueError(f"Unsupported vector index provider: {config.provider}")
