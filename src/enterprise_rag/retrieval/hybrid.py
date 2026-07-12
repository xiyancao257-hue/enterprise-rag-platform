from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.embeddings.base import EmbeddingModel
from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.retrieval.bm25 import BM25Retriever
from enterprise_rag.retrieval.filters import MetadataFilter
from enterprise_rag.retrieval.vector import HashingVectorRetriever
from enterprise_rag.vector_index.base import VectorIndex


class SingleQueryRetriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Search a single query and return ranked hits."""


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        extra_retrievers: list[SingleQueryRetriever] | None = None,
        vector_index: VectorIndex | None = None,
        embedding_model: EmbeddingModel | None = None,
        embedding_cache: CacheStore | None = None,
        embedding_ttl_seconds: int | None = 86_400,
    ) -> None:
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.vector = HashingVectorRetriever(
            chunks,
            embedding_model=embedding_model,
            vector_index=vector_index,
            embedding_cache=embedding_cache,
            embedding_ttl_seconds=embedding_ttl_seconds,
        )
        self.extra_retrievers = extra_retrievers or []

    def search(
        self,
        queries: list[str],
        top_k: int = 8,
        metadata_filters: dict[str, str] | None = None,
        user_groups: set[str] | None = None,
        user_id: str | None = None,
        user_roles: set[str] | None = None,
    ) -> list[SearchHit]:
        metadata_filter = MetadataFilter(
            metadata_filters,
            user_groups=user_groups,
            user_id=user_id,
            user_roles=user_roles,
        )
        candidate_lists: list[list[SearchHit]] = []
        for query in queries:
            candidate_lists.append(metadata_filter.apply_hits(self.bm25.search(query, top_k=top_k * 2)))
            candidate_lists.append(
                metadata_filter.apply_hits(
                    self.vector.search(query, top_k=top_k * 2, metadata_filters=metadata_filters)
                )
            )
            for retriever in self.extra_retrievers:
                candidate_lists.append(metadata_filter.apply_hits(retriever.search(query, top_k=top_k * 2)))
        return self._rrf(candidate_lists, top_k=top_k)

    def _rrf(self, candidate_lists: list[list[SearchHit]], top_k: int, k: int = 60) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, Chunk] = {}
        retrievers: dict[str, set[str]] = defaultdict(set)

        for hits in candidate_lists:
            for hit in hits:
                scores[hit.chunk.id] += 1 / (k + hit.rank)
                chunks_by_id[hit.chunk.id] = hit.chunk
                retrievers[hit.chunk.id].add(hit.retriever)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchHit(
                chunk=chunks_by_id[chunk_id],
                score=score,
                retriever="+".join(sorted(retrievers[chunk_id])),
                rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]
