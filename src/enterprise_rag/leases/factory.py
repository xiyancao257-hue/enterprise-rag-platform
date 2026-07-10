from __future__ import annotations

from enterprise_rag.config import LeaseConfig
from enterprise_rag.leases.base import LeaseStore
from enterprise_rag.leases.in_memory import InMemoryLeaseStore
from enterprise_rag.leases.redis_lease import RedisLeaseStore


def create_lease_store(config: LeaseConfig) -> LeaseStore:
    provider = config.provider.lower()
    if provider == "memory":
        return InMemoryLeaseStore()
    if provider == "redis":
        return RedisLeaseStore(url=config.url, prefix=config.prefix)
    raise ValueError(f"Unsupported lease provider: {config.provider}")
