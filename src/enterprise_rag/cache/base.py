from __future__ import annotations

from typing import Protocol


class CacheStore(Protocol):
    def get(self, key: str) -> object | None:
        """Return a cached value or None when absent/expired."""

    def set(self, key: str, value: object, ttl_seconds: int | None = None) -> None:
        """Store a cache value with an optional TTL."""
