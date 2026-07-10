from __future__ import annotations

import json
from typing import Any


class RedisCacheNotConfiguredError(RuntimeError):
    pass


class RedisCache:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "enterprise-rag",
        client: Any | None = None,
    ) -> None:
        self.prefix = prefix.rstrip(":")
        self.client = client or self._create_client(url)

    def get(self, key: str) -> object | None:
        raw = self.client.get(self._key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def set(self, key: str, value: object, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        namespaced_key = self._key(key)
        if ttl_seconds is not None and ttl_seconds > 0:
            self.client.setex(namespaced_key, ttl_seconds, payload)
        else:
            self.client.set(namespaced_key, payload)

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def _create_client(self, url: str) -> Any:
        try:
            import redis
        except ImportError as exc:
            raise RedisCacheNotConfiguredError(
                "Redis cache support requires the optional dependency: install enterprise-rag[redis]."
            ) from exc
        return redis.Redis.from_url(url)
