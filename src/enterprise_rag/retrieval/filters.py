from __future__ import annotations

from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.security.access_control import AccessContext, AccessPolicy


class MetadataFilter:
    def __init__(
        self,
        filters: dict[str, str] | None = None,
        user_groups: set[str] | None = None,
        user_id: str | None = None,
        user_roles: set[str] | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> None:
        self.filters = filters or {}
        self.access_context = AccessContext(
            tenant_id=self.filters.get("tenant_id"),
            user_id=user_id,
            groups=user_groups or set(),
            roles=user_roles or set(),
        )
        self.access_policy = access_policy or AccessPolicy()

    def matches(self, chunk: Chunk) -> bool:
        for key, expected in self.filters.items():
            if chunk.metadata.get(key) != expected:
                return False
        return self.access_policy.can_read(chunk, self.access_context)

    def apply_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [chunk for chunk in chunks if self.matches(chunk)]

    def apply_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        return [hit for hit in hits if self.matches(hit.chunk)]
