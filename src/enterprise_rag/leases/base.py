from __future__ import annotations

from typing import Protocol


class LeaseStore(Protocol):
    def acquire(self, name: str, owner: str, ttl_seconds: int) -> bool:
        """Try to acquire a named lease for one owner."""

    def release(self, name: str, owner: str) -> bool:
        """Release a named lease only if it is still owned by this owner."""

    def get_owner(self, name: str) -> str | None:
        """Return the current lease owner, if any."""
