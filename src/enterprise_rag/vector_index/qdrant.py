from __future__ import annotations

from typing import Any, Protocol

from enterprise_rag.vector_index.base import VectorSearchResult


class QdrantClientLike(Protocol):
    def upsert(self, collection_name: str, points: list[Any]) -> Any:
        """Insert or replace points in a Qdrant collection."""

    def query_points(self, collection_name: str, query: list[float], limit: int) -> Any:
        """Query nearest points in a Qdrant collection."""


class QdrantVectorIndex:
    def __init__(
        self,
        collection_name: str,
        client: QdrantClientLike | None = None,
        point_factory: Any | None = None,
        vector_params_factory: Any | None = None,
        url: str = "http://localhost:6333",
    ) -> None:
        self.collection_name = collection_name
        self.client = client or self._build_client(url)
        self.point_factory = point_factory or self._point_struct
        self.vector_params_factory = vector_params_factory or self._vector_params
        self._collection_ready = False

    def add(self, id: str, vector: list[float]) -> None:
        self._ensure_collection(vector_size=len(vector))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[self.point_factory(id, vector)],
        )

    def search(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        points = getattr(response, "points", response)
        return [
            VectorSearchResult(id=str(point.id), score=float(point.score), rank=rank)
            for rank, point in enumerate(points, start=1)
        ]

    def _build_client(self, url: str) -> QdrantClientLike:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return QdrantClient(url=url)

    def _point_struct(self, id: str, vector: list[float]) -> Any:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return PointStruct(id=id, vector=vector)

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return

        collection_exists = getattr(self.client, "collection_exists", None)
        if collection_exists is None:
            self._collection_ready = True
            return

        if not collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_params_factory(vector_size),
            )
        self._collection_ready = True

    def _vector_params(self, vector_size: int) -> Any:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return VectorParams(size=vector_size, distance=Distance.COSINE)
