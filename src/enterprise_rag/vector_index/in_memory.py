from __future__ import annotations

from enterprise_rag.vector_index.base import VectorSearchResult


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def add(self, id: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        self.vectors[id] = vector
        self.metadata[id] = metadata or {}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        scored = []
        for id, vector in self.vectors.items():
            if not self._matches_metadata(id, metadata_filters):
                continue
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

    def _matches_metadata(self, id: str, metadata_filters: dict[str, str] | None) -> bool:
        if not metadata_filters:
            return True
        metadata = self.metadata.get(id, {})
        return all(metadata.get(key) == expected for key, expected in metadata_filters.items())
