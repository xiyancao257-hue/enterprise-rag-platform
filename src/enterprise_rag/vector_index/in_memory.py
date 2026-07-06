from __future__ import annotations

from enterprise_rag.vector_index.base import VectorSearchResult


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def add(self, id: str, vector: list[float]) -> None:
        self.vectors[id] = vector

    def search(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        scored = []
        for id, vector in self.vectors.items():
            score = self._cosine(query_vector, vector)
            if score > 0:
                scored.append((score, id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            VectorSearchResult(id=id, score=score, rank=rank)
            for rank, (score, id) in enumerate(scored[:top_k], start=1)
        ]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))
