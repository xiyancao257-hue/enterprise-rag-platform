from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheEntry:
    value: object
    expires_at: float | None = None


class InMemoryCache:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self.now = now or time.time
        self.entries: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> object | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at is not None and entry.expires_at <= self.now():
            self.entries.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def set(self, key: str, value: object, ttl_seconds: int | None = None) -> None:
        expires_at = self.now() + ttl_seconds if ttl_seconds is not None and ttl_seconds > 0 else None
        self.entries[key] = CacheEntry(value=value, expires_at=expires_at)
