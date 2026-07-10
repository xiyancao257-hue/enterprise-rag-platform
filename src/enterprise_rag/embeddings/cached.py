from __future__ import annotations

import hashlib

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.embeddings.base import EmbeddingModel


class CachedEmbeddingModel:
    def __init__(
        self,
        model: EmbeddingModel,
        cache: CacheStore | None = None,
        model_id: str | None = None,
        ttl_seconds: int | None = 86_400,
    ) -> None:
        self.model = model
        self.cache = cache or InMemoryCache()
        self.model_id = model_id or model.__class__.__name__
        self.ttl_seconds = ttl_seconds

    def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        vector = self.model.embed(text)
        self.cache.set(key, list(vector), ttl_seconds=self.ttl_seconds)
        return vector

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{self.model_id}:{digest}"
