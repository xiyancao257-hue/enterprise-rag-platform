from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Lease:
    owner: str
    expires_at: float


class InMemoryLeaseStore:
    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self.now = now or time.time
        self.leases: dict[str, _Lease] = {}

    def acquire(self, name: str, owner: str, ttl_seconds: int) -> bool:
        now = self.now()
        lease = self.leases.get(name)
        if lease is not None and lease.expires_at > now:
            return False
        self.leases[name] = _Lease(owner=owner, expires_at=now + max(1, ttl_seconds))
        return True

    def release(self, name: str, owner: str) -> bool:
        lease = self.leases.get(name)
        if lease is None or lease.owner != owner:
            return False
        del self.leases[name]
        return True

    def get_owner(self, name: str) -> str | None:
        lease = self.leases.get(name)
        if lease is None:
            return None
        if lease.expires_at <= self.now():
            del self.leases[name]
            return None
        return lease.owner
