from __future__ import annotations

from typing import Any, Protocol

from enterprise_rag.vector_index.base import VectorSearchResult


class QdrantClientLike(Protocol):
    def upsert(self, collection_name: str, points: list[Any]) -> Any:
        """Insert or replace points in a Qdrant collection."""

    def delete(self, collection_name: str, points_selector: Any) -> Any:
        """Delete points from a Qdrant collection."""

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> Any:
        """Query nearest points in a Qdrant collection."""


class QdrantVectorIndex:
    def __init__(
        self,
        collection_name: str,
        client: QdrantClientLike | None = None,
        point_factory: Any | None = None,
        vector_params_factory: Any | None = None,
        filter_factory: Any | None = None,
        delete_selector_factory: Any | None = None,
        url: str = "http://localhost:6333",
    ) -> None:
        self.collection_name = collection_name
        self.client = client or self._build_client(url)
        self.point_factory = point_factory or self._point_struct
        self.vector_params_factory = vector_params_factory or self._vector_params
        self.filter_factory = filter_factory or self._payload_filter
        self.delete_selector_factory = delete_selector_factory or self._point_ids_selector
        self._collection_ready = False

    def add(self, id: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        self._ensure_collection(vector_size=len(vector))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[self.point_factory(id, vector, metadata or {})],
        )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.delete_selector_factory(ids),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=self.filter_factory(metadata_filters or {}),
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

    def _point_struct(self, id: str, vector: list[float], metadata: dict[str, str]) -> Any:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return PointStruct(id=id, vector=vector, payload=metadata)

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

    def _payload_filter(self, metadata_filters: dict[str, str]) -> Any | None:
        if not metadata_filters:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadata_filters.items()]
        )

    def _point_ids_selector(self, ids: list[str]) -> Any:
        try:
            from qdrant_client.models import PointIdsList
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant support requires the optional dependency: install enterprise-rag[qdrant]."
            ) from exc
        return PointIdsList(points=ids)
