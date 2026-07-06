from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    rank: int


class VectorIndex(Protocol):
    def add(self, id: str, vector: list[float]) -> None:
        """Add or replace a vector by id."""

    def search(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        """Search for nearest vectors."""
