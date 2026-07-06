from __future__ import annotations

from enterprise_rag.models import SearchHit
from enterprise_rag.text import jaccard, tokenize


class LightweightReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        query_terms = tokenize(query)
        rescored = []
        for hit in hits:
            lexical_overlap = jaccard(query_terms, tokenize(hit.chunk.text))
            heading_bonus = 0.05 if any(term in " ".join(hit.chunk.heading_path).lower() for term in query_terms) else 0.0
            score = hit.score + lexical_overlap + heading_bonus
            rescored.append((score, hit))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(chunk=hit.chunk, score=score, retriever=f"{hit.retriever}+rerank", rank=rank)
            for rank, (score, hit) in enumerate(rescored[:top_k], start=1)
        ]

