from __future__ import annotations

from enterprise_rag.models import SearchHit
from enterprise_rag.text import tokenize


class ContextCompressor:
    def compress(self, query: str, hits: list[SearchHit], max_sentences_per_hit: int = 3) -> list[SearchHit]:
        query_terms = set(tokenize(query))
        compressed = []
        for hit in hits:
            sentences = self._split_sentences(hit.chunk.text)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(query_terms & set(tokenize(sentence))),
                reverse=True,
            )
            selected = [sentence for sentence in ranked[:max_sentences_per_hit] if sentence.strip()]
            if not selected:
                selected = sentences[:max_sentences_per_hit]
            chunk = hit.chunk.__class__(
                id=hit.chunk.id,
                document_id=hit.chunk.document_id,
                text=" ".join(selected),
                heading_path=hit.chunk.heading_path,
                source_blocks=hit.chunk.source_blocks,
                metadata=hit.chunk.metadata,
            )
            compressed.append(hit.__class__(chunk=chunk, score=hit.score, retriever=hit.retriever, rank=hit.rank))
        return compressed

    def _split_sentences(self, text: str) -> list[str]:
        normalized = text.replace("\n", " ")
        parts = []
        start = 0
        for index, char in enumerate(normalized):
            if char in ".!?":
                parts.append(normalized[start : index + 1].strip())
                start = index + 1
        tail = normalized[start:].strip()
        if tail:
            parts.append(tail)
        return parts or [normalized]

