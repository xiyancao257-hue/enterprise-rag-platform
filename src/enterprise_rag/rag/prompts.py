from __future__ import annotations

from enterprise_rag.models import SearchHit
from enterprise_rag.rag.citations import CitationFormatter


class GroundedQAPromptTemplate:
    def __init__(self, citation_formatter: CitationFormatter | None = None) -> None:
        self.citation_formatter = citation_formatter or CitationFormatter()

    def render(self, question: str, hits: list[SearchHit]) -> str:
        evidence = self._format_evidence(hits)
        citations = "\n".join(self.citation_formatter.format_many(hits)) if hits else "No citations."
        return (
            "You are an enterprise RAG assistant.\n\n"
            "Answer the user question using only the provided evidence.\n"
            "If the evidence is insufficient, say you do not have enough information.\n"
            "Do not invent facts, sources, policies, numbers, or citations.\n"
            "Cite every factual claim with bracket citations like [1] or [2].\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Citations:\n{citations}\n\n"
            "Answer:"
        )

    def _format_evidence(self, hits: list[SearchHit]) -> str:
        if not hits:
            return "No evidence was retrieved."
        return "\n\n".join(f"[{hit.rank}] {hit.chunk.text}" for hit in hits)
