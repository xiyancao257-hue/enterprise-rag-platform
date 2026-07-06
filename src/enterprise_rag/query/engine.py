from __future__ import annotations

import re
from difflib import get_close_matches

from enterprise_rag.models import QueryPlan
from enterprise_rag.text import tokenize

FILTER_RE = re.compile(r"\b(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\:(?P<value>[a-zA-Z0-9_\-\.]+)\b")


class QueryEngine:
    def __init__(self, vocabulary: set[str] | None = None) -> None:
        self.vocabulary = vocabulary or set()

    def plan(self, query: str) -> QueryPlan:
        normalized = " ".join(query.strip().split())
        filter_query, metadata_filters = self._extract_metadata_filters(normalized)
        corrections = self._correct_typos(filter_query)
        corrected = self._apply_corrections(filter_query, corrections)
        rewritten = self._rewrite(corrected)
        ambiguity_notes = self._detect_ambiguity(corrected)
        return QueryPlan(
            original_query=query,
            normalized_query=corrected,
            rewritten_queries=tuple(dict.fromkeys([corrected, *rewritten])),
            ambiguity_notes=tuple(ambiguity_notes),
            corrections=corrections,
            metadata_filters=metadata_filters,
        )

    def _extract_metadata_filters(self, query: str) -> tuple[str, dict[str, str]]:
        filters = {match.group("key"): match.group("value") for match in FILTER_RE.finditer(query)}
        cleaned = FILTER_RE.sub("", query)
        return " ".join(cleaned.split()), filters

    def _correct_typos(self, query: str) -> dict[str, str]:
        if not self.vocabulary:
            return {}
        corrections = {}
        for token in tokenize(query):
            if token in self.vocabulary or len(token) < 5:
                continue
            matches = get_close_matches(token, self.vocabulary, n=1, cutoff=0.84)
            if matches:
                corrections[token] = matches[0]
        return corrections

    def _apply_corrections(self, query: str, corrections: dict[str, str]) -> str:
        words = query.split()
        return " ".join(corrections.get(word.lower(), word) for word in words)

    def _rewrite(self, query: str) -> list[str]:
        rewrites = []
        lower = query.lower()
        if " vs " in lower or " versus " in lower:
            rewrites.append(query.replace(" vs ", " compare ").replace(" versus ", " compare "))
        if lower.startswith("how to "):
            rewrites.append(query[7:])
        if lower.startswith("what is "):
            rewrites.append(query[8:])
        return rewrites

    def _detect_ambiguity(self, query: str) -> list[str]:
        notes = []
        vague_terms = {"it", "this", "that", "they", "them", "there"}
        if any(token in vague_terms for token in tokenize(query)):
            notes.append("Query contains references that may need conversation history or clarification.")
        if len(tokenize(query)) <= 2:
            notes.append("Query is very short; expansion or clarification may improve recall.")
        return notes
