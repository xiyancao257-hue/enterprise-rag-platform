from __future__ import annotations

from enterprise_rag.config import EmbeddingConfig
from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.embeddings.openai_embeddings import OpenAIEmbeddingModel
from enterprise_rag.providers.resilience import ProviderResiliencePolicy


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    provider = config.provider.lower()
    if provider == "hashing":
        return HashingEmbeddingModel(dimensions=config.dimensions)
    if provider == "openai":
        return OpenAIEmbeddingModel(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            resilience=ProviderResiliencePolicy(
                max_retries=config.max_retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_reset_seconds=config.circuit_breaker_reset_seconds,
            ),
        )
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
