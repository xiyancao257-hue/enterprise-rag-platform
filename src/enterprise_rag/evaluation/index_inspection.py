from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.models import Chunk
from enterprise_rag.text import tokenize


@dataclass(frozen=True)
class IndexQualityReport:
    chunk_count: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    empty_chunks: int
    chunks_missing_source: int
    chunks_missing_extension: int
    acl_protected_chunks: int
    metadata_keys: tuple[str, ...]


def inspect_index(chunks: list[Chunk]) -> IndexQualityReport:
    token_counts = [len(tokenize(chunk.text)) for chunk in chunks]
    metadata_keys = sorted({key for chunk in chunks for key in chunk.metadata})
    return IndexQualityReport(
        chunk_count=len(chunks),
        avg_tokens=sum(token_counts) / len(token_counts) if token_counts else 0.0,
        min_tokens=min(token_counts) if token_counts else 0,
        max_tokens=max(token_counts) if token_counts else 0,
        empty_chunks=sum(1 for chunk in chunks if not chunk.text.strip()),
        chunks_missing_source=sum(1 for chunk in chunks if not chunk.metadata.get("source_path")),
        chunks_missing_extension=sum(1 for chunk in chunks if not chunk.metadata.get("extension")),
        acl_protected_chunks=sum(1 for chunk in chunks if chunk.metadata.get("allowed_groups")),
        metadata_keys=tuple(metadata_keys),
    )


def format_index_quality_report(report: IndexQualityReport) -> str:
    return "\n".join(
        [
            "Index Quality Report",
            f"chunks: {report.chunk_count}",
            f"avg_tokens: {report.avg_tokens:.1f}",
            f"min_tokens: {report.min_tokens}",
            f"max_tokens: {report.max_tokens}",
            f"empty_chunks: {report.empty_chunks}",
            f"chunks_missing_source: {report.chunks_missing_source}",
            f"chunks_missing_extension: {report.chunks_missing_extension}",
            f"acl_protected_chunks: {report.acl_protected_chunks}",
            f"metadata_keys: {', '.join(report.metadata_keys) if report.metadata_keys else 'none'}",
        ]
    )

