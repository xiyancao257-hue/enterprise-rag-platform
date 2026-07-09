from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from enterprise_rag.ingestion.loaders import load_documents
from enterprise_rag.models import Chunk, Document
from enterprise_rag.processing.chunking import StructureAwareChunker
from enterprise_rag.processing.cleaning import DirtyDataCleaner
from enterprise_rag.processing.parser import StructureParser
from enterprise_rag.storage.json_store import JsonChunkStore


@dataclass(frozen=True)
class IngestReport:
    documents_loaded: int
    documents_new: int
    documents_updated: int
    documents_unchanged: int
    documents_deleted: int
    documents_filtered: int
    chunks_indexed: int
    chunks_upserted: tuple[str, ...] = ()
    chunks_deleted: tuple[str, ...] = ()


class IncrementalIngestPipeline:
    def __init__(
        self,
        cleaner: DirtyDataCleaner | None = None,
        parser: StructureParser | None = None,
        chunker: StructureAwareChunker | None = None,
    ) -> None:
        self.cleaner = cleaner or DirtyDataCleaner()
        self.parser = parser or StructureParser()
        self.chunker = chunker or StructureAwareChunker()

    def run(
        self,
        source_path: Path,
        store: JsonChunkStore,
        metadata_overrides: dict[str, str] | None = None,
    ) -> IngestReport:
        metadata_overrides = metadata_overrides or {}
        ingest_scope = metadata_overrides.get("tenant_id", "")
        existing_chunks = store.load()
        existing_by_source = self._group_by_source(existing_chunks)
        documents = [
            self._with_metadata_overrides(document, metadata_overrides) for document in load_documents(source_path)
        ]
        current_sources = {self._source_key(document.metadata, document.source_path) for document in documents}

        next_chunks = [
            chunk
            for source_key, chunks in existing_by_source.items()
            if source_key[0] != ingest_scope
            for chunk in chunks
        ]
        documents_new = 0
        documents_updated = 0
        documents_unchanged = 0
        documents_filtered = 0
        chunks_upserted: list[str] = []
        chunks_deleted: list[str] = []

        for document in documents:
            previous_chunks = existing_by_source.get(self._source_key(document.metadata, document.source_path), [])
            if previous_chunks and self._content_hash(previous_chunks) == document.metadata.get("content_hash"):
                next_chunks.extend(previous_chunks)
                documents_unchanged += 1
                continue

            processed_chunks = self._process_document(document)
            if not processed_chunks:
                documents_filtered += 1
                chunks_deleted.extend(chunk.id for chunk in previous_chunks)
            elif previous_chunks:
                documents_updated += 1
                chunks_deleted.extend(chunk.id for chunk in previous_chunks)
                chunks_upserted.extend(chunk.id for chunk in processed_chunks)
            else:
                documents_new += 1
                chunks_upserted.extend(chunk.id for chunk in processed_chunks)

            next_chunks.extend(processed_chunks)

        existing_sources_in_scope = {source_key for source_key in existing_by_source if source_key[0] == ingest_scope}
        deleted_sources = existing_sources_in_scope - current_sources
        for source in deleted_sources:
            chunks_deleted.extend(chunk.id for chunk in existing_by_source[source])
        documents_deleted = len(deleted_sources)
        store.save(next_chunks)
        return IngestReport(
            documents_loaded=len(documents),
            documents_new=documents_new,
            documents_updated=documents_updated,
            documents_unchanged=documents_unchanged,
            documents_deleted=documents_deleted,
            documents_filtered=documents_filtered,
            chunks_indexed=len(next_chunks),
            chunks_upserted=tuple(chunks_upserted),
            chunks_deleted=tuple(chunks_deleted),
        )

    def _process_document(self, document: Document) -> list[Chunk]:
        cleaned = self.cleaner.clean(document)
        if cleaned is None:
            return []
        blocks = self.parser.parse(cleaned)
        return self.chunker.chunk(blocks)

    def _group_by_source(self, chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        grouped: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in chunks:
            source_path = chunk.metadata.get("source_path")
            if source_path is None:
                continue
            grouped.setdefault(self._source_key(chunk.metadata, source_path), []).append(chunk)
        return grouped

    def _content_hash(self, chunks: list[Chunk]) -> str | None:
        for chunk in chunks:
            content_hash = chunk.metadata.get("content_hash")
            if content_hash:
                return content_hash
        return None

    def _with_metadata_overrides(self, document: Document, metadata_overrides: dict[str, str]) -> Document:
        if not metadata_overrides:
            return document
        return replace(document, metadata={**document.metadata, **metadata_overrides})

    def _source_key(self, metadata: dict[str, str], source_path: str) -> tuple[str, str]:
        return (metadata.get("tenant_id", ""), source_path)
