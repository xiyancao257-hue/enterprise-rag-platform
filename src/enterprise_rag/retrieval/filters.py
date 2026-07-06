from __future__ import annotations

from enterprise_rag.models import Chunk, SearchHit


class MetadataFilter:
    def __init__(self, filters: dict[str, str] | None = None, user_groups: set[str] | None = None) -> None:
        self.filters = filters or {}
        self.user_groups = user_groups or set()

    def matches(self, chunk: Chunk) -> bool:
        for key, expected in self.filters.items():
            if chunk.metadata.get(key) != expected:
                return False
        return self._matches_acl(chunk)

    def _matches_acl(self, chunk: Chunk) -> bool:
        allowed_groups = self._parse_groups(chunk.metadata.get("allowed_groups", ""))
        if not allowed_groups:
            return True
        if not self.user_groups:
            return False
        return bool(allowed_groups & self.user_groups)

    def _parse_groups(self, value: str) -> set[str]:
        return {group.strip() for group in value.split(",") if group.strip()}

    def apply_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not self.filters and not self.user_groups:
            return chunks
        return [chunk for chunk in chunks if self.matches(chunk)]

    def apply_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        if not self.filters and not self.user_groups:
            return hits
        return [hit for hit in hits if self.matches(hit.chunk)]
