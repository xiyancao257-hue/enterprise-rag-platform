from __future__ import annotations

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.cache.redis_cache import RedisCache
from enterprise_rag.config import CacheConfig


def create_cache(config: CacheConfig) -> CacheStore:
    provider = config.provider.lower()
    if provider == "memory":
        return InMemoryCache()
    if provider == "redis":
        return RedisCache(url=config.url, prefix=config.prefix)
    raise ValueError(f"Unsupported cache provider: {config.provider}")
