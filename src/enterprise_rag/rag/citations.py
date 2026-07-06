from __future__ import annotations

from enterprise_rag.models import SearchHit


class CitationFormatter:
    def format(self, hit: SearchHit) -> str:
        source = hit.chunk.metadata.get("source_path") or "unknown"
        heading = self._format_heading(hit)
        location = f"{source} - {heading}" if heading else source
        return (
            f"[{hit.rank}] {location} "
            f"(chunk={hit.chunk.id}, score={hit.score:.4f}, retriever={hit.retriever})"
        )

    def format_many(self, hits: tuple[SearchHit, ...] | list[SearchHit]) -> list[str]:
        return [self.format(hit) for hit in hits]

    def _format_heading(self, hit: SearchHit) -> str:
        return " > ".join(part for part in hit.chunk.heading_path if part)

