from __future__ import annotations

from enterprise_rag.models import SearchHit


class ExternalRerankerNotConfiguredError(RuntimeError):
    pass


class ExternalReranker:
    def __init__(self, provider: str = "external") -> None:
        self.provider = provider

    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        raise ExternalRerankerNotConfiguredError(
            "ExternalReranker is an adapter stub. Configure a reranking provider before using it."
        )
