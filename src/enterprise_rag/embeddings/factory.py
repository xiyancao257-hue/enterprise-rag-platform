from __future__ import annotations

from enterprise_rag.config import EmbeddingConfig
from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.embeddings.openai_embeddings import OpenAIEmbeddingModel


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    provider = config.provider.lower()
    if provider == "hashing":
        return HashingEmbeddingModel(dimensions=config.dimensions)
    if provider == "openai":
        return OpenAIEmbeddingModel(model=config.model)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
