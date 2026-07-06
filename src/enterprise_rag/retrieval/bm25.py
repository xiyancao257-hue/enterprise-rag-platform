from __future__ import annotations

import math
from collections import Counter

from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.text import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs = Counter(term for tokens in self.doc_tokens for term in set(tokens))
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query_terms = tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for chunk, terms, freqs in zip(self.chunks, self.doc_tokens, self.term_freqs):
            score = 0.0
            doc_len = len(terms)
            for term in query_terms:
                if term not in freqs:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [SearchHit(chunk=chunk, score=score, retriever="bm25", rank=i + 1) for i, (score, chunk) in enumerate(scored[:top_k])]

