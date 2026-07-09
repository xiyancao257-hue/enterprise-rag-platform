from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.models import SearchHit


@dataclass(frozen=True)
class TraceHit:
    chunk_id: str
    document_id: str
    score: float
    retriever: str
    rank: int
    heading_path: tuple[str, ...]
    source_path: str | None = None


@dataclass(frozen=True)
class QueryTrace:
    original_query: str
    normalized_query: str
    rewritten_queries: tuple[str, ...]
    metadata_filters: dict[str, str]
    retrieved: tuple[TraceHit, ...] = ()
    reranked: tuple[TraceHit, ...] = ()
    blocked_context: tuple[TraceHit, ...] = ()
    final_context: tuple[TraceHit, ...] = ()


def trace_hits(hits: list[SearchHit] | tuple[SearchHit, ...]) -> tuple[TraceHit, ...]:
    return tuple(
        TraceHit(
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            score=hit.score,
            retriever=hit.retriever,
            rank=hit.rank,
            heading_path=hit.chunk.heading_path,
            source_path=hit.chunk.metadata.get("source_path"),
        )
        for hit in hits
    )


def format_query_trace(trace: QueryTrace) -> str:
    lines = [
        "Trace",
        f"- original query: {trace.original_query}",
        f"- normalized query: {trace.normalized_query}",
        f"- rewritten queries: {', '.join(trace.rewritten_queries)}",
    ]
    if trace.metadata_filters:
        lines.append(f"- metadata filters: {trace.metadata_filters}")

    lines.extend(
        [
            "",
            _format_stage("Retrieved", trace.retrieved),
            "",
            _format_stage("Reranked", trace.reranked),
            "",
            _format_stage("Blocked context", trace.blocked_context),
            "",
            _format_stage("Final context", trace.final_context),
        ]
    )
    return "\n".join(lines)


def _format_stage(title: str, hits: tuple[TraceHit, ...]) -> str:
    if not hits:
        return f"{title}\n- no hits"

    lines = [title]
    for hit in hits:
        heading = " > ".join(hit.heading_path) if hit.heading_path else "(no heading)"
        lines.append(
            f"- rank={hit.rank} score={hit.score:.4f} retriever={hit.retriever} "
            f"chunk={hit.chunk_id} source={hit.source_path or hit.document_id} heading={heading}"
        )
    return "\n".join(lines)
