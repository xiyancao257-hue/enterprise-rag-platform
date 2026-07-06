from __future__ import annotations

from typing import Protocol

from enterprise_rag.models import SearchHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        """Rerank retrieved hits for a query."""
