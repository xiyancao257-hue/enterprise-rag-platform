from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from enterprise_rag.ingestion.loaders import FilteredDocument, load_documents_with_report
from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.models import Chunk, Document
from enterprise_rag.processing.chunking import StructureAwareChunker
from enterprise_rag.processing.cleaning import DirtyDataCleaner
from enterprise_rag.processing.parser import StructureParser
from enterprise_rag.processing.redaction import SensitiveDataRedactor
from enterprise_rag.storage.json_store import JsonChunkStore

FILTER_LOW_QUALITY_TEXT = "low_quality_text"


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
    filter_reasons: dict[str, int] = field(default_factory=dict)
    filtered_documents: tuple[FilteredDocument, ...] = ()
    dry_run: bool = False


class IncrementalIngestPipeline:
    def __init__(
        self,
        cleaner: DirtyDataCleaner | None = None,
        redactor: SensitiveDataRedactor | None = None,
        parser: StructureParser | None = None,
        chunker: StructureAwareChunker | None = None,
        file_policy: IngestionFilePolicy | None = None,
    ) -> None:
        self.cleaner = cleaner or DirtyDataCleaner()
        self.redactor = redactor or SensitiveDataRedactor()
        self.parser = parser or StructureParser()
        self.chunker = chunker or StructureAwareChunker()
        self.file_policy = file_policy or IngestionFilePolicy()

    def run(
        self,
        source_path: Path,
        store: JsonChunkStore,
        metadata_overrides: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> IngestReport:
        metadata_overrides = metadata_overrides or {}
        ingest_scope = metadata_overrides.get("tenant_id", "")
        existing_chunks = store.load()
        existing_by_source = self._group_by_source(existing_chunks)
        load_result = load_documents_with_report(source_path, policy=self.file_policy)
        documents = [self._with_metadata_overrides(document, metadata_overrides) for document in load_result.documents]
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
        documents_filtered = load_result.documents_filtered
        filter_reasons = dict(load_result.filter_reasons)
        filtered_documents = list(load_result.filtered_documents)
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
                self._count_filter_reason(filter_reasons, FILTER_LOW_QUALITY_TEXT)
                filtered_documents.append(
                    FilteredDocument(source_path=document.source_path, reason=FILTER_LOW_QUALITY_TEXT)
                )
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
        if not dry_run:
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
            filter_reasons=filter_reasons,
            filtered_documents=tuple(filtered_documents),
            dry_run=dry_run,
        )

    def _process_document(self, document: Document) -> list[Chunk]:
        cleaned = self.cleaner.clean(document)
        if cleaned is None:
            return []
        redacted = self.redactor.redact(cleaned)
        blocks = self.parser.parse(redacted)
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

    def _count_filter_reason(self, filter_reasons: dict[str, int], reason: str) -> None:
        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
