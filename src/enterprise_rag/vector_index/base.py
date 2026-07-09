from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    rank: int


class VectorIndex(Protocol):
    def add(self, id: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        """Add or replace a vector by id."""

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by id."""

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for nearest vectors."""
