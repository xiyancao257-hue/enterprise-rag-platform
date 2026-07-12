from pathlib import Path

from enterprise_rag.config import ChunkingConfig, ChunkingProfileConfig
from enterprise_rag.ingestion.loaders import load_documents, load_documents_with_report
from enterprise_rag.ingestion.pipeline import FILTER_LOW_QUALITY_TEXT, IncrementalIngestPipeline
from enterprise_rag.ingestion.policy import FILTER_FILE_TOO_LARGE, FILTER_UNSUPPORTED_EXTENSION, IngestionFilePolicy
from enterprise_rag.models import BlockType, Document
from enterprise_rag.processing.chunking import StructureAwareChunker
from enterprise_rag.processing.cleaning import DirtyDataCleaner
from enterprise_rag.processing.parser import StructureParser
from enterprise_rag.processing.redaction import SensitiveDataRedactor
from enterprise_rag.storage.json_store import JsonChunkStore


def test_load_documents_reads_supported_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    markdown = raw_dir / "guide.md"
    ignored = raw_dir / "image.png"
    markdown.write_text("# Guide\n\nHybrid retrieval notes.", encoding="utf-8")
    ignored.write_text("not a document", encoding="utf-8")

    documents = load_documents(raw_dir)

    assert len(documents) == 1
    assert documents[0].source_path == str(markdown)
    assert documents[0].metadata["extension"] == ".md"
    assert documents[0].metadata["filename"] == "guide.md"
    assert documents[0].metadata["content_hash"]


def test_load_documents_converts_csv_to_markdown_table(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "incidents.csv"
    csv_file.write_text(
        "service,incident,severity\nAuth Service,AUTH-429,high\nBilling,BILL-204,medium\n",
        encoding="utf-8",
    )

    documents = load_documents(raw_dir)

    assert len(documents) == 1
    assert documents[0].metadata["extension"] == ".csv"
    assert documents[0].metadata["source_format"] == "csv"
    assert documents[0].metadata["table_format"] == "markdown"
    assert "# Incidents" in documents[0].text
    assert "| service | incident | severity |" in documents[0].text
    assert "| Auth Service | AUTH-429 | high |" in documents[0].text


def test_load_documents_report_counts_policy_filtered_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text("# Guide\n\nHybrid retrieval notes.", encoding="utf-8")
    raw_dir.joinpath("image.png").write_text("not a supported document", encoding="utf-8")
    raw_dir.joinpath("large.txt").write_text("x" * 80, encoding="utf-8")

    result = load_documents_with_report(
        raw_dir,
        policy=IngestionFilePolicy(allowed_extensions=(".md", ".txt"), max_file_bytes=50),
    )

    assert len(result.documents) == 1
    assert result.documents[0].metadata["filename"] == "guide.md"
    assert result.documents_filtered == 2
    assert result.filter_reasons == {
        FILTER_UNSUPPORTED_EXTENSION: 1,
        FILTER_FILE_TOO_LARGE: 1,
    }
    assert [(item.source_path, item.reason) for item in result.filtered_documents] == [
        (str(raw_dir / "image.png"), FILTER_UNSUPPORTED_EXTENSION),
        (str(raw_dir / "large.txt"), FILTER_FILE_TOO_LARGE),
    ]


def test_incremental_ingest_counts_file_policy_filtered_documents(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    raw_dir.joinpath("diagram.png").write_text("not a supported document", encoding="utf-8")
    raw_dir.joinpath("oversized.txt").write_text("x" * 150, encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")

    report = IncrementalIngestPipeline(
        file_policy=IngestionFilePolicy(allowed_extensions=(".md", ".txt"), max_file_bytes=100)
    ).run(raw_dir, store)

    assert report.documents_loaded == 1
    assert report.documents_filtered == 2
    assert report.filter_reasons == {
        FILTER_UNSUPPORTED_EXTENSION: 1,
        FILTER_FILE_TOO_LARGE: 1,
    }
    assert [(item.source_path, item.reason) for item in report.filtered_documents] == [
        (str(raw_dir / "diagram.png"), FILTER_UNSUPPORTED_EXTENSION),
        (str(raw_dir / "oversized.txt"), FILTER_FILE_TOO_LARGE),
    ]
    assert report.documents_new == 1


def test_incremental_ingest_records_cleaner_filter_reason(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("tiny.md").write_text("short", encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")

    report = IncrementalIngestPipeline().run(raw_dir, store)

    assert report.documents_loaded == 1
    assert report.documents_filtered == 1
    assert report.filter_reasons == {FILTER_LOW_QUALITY_TEXT: 1}
    assert [(item.source_path, item.reason) for item in report.filtered_documents] == [
        (str(raw_dir / "tiny.md"), FILTER_LOW_QUALITY_TEXT)
    ]
    assert store.load() == []


def test_incremental_ingest_uses_chunking_config_by_extension(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    raw_dir.joinpath("notes.txt").write_text(
        "Plain text notes describe BM25 keyword recall and vector search for enterprise retrieval.",
        encoding="utf-8",
    )
    store = JsonChunkStore(tmp_path / "chunks.json")
    pipeline = IncrementalIngestPipeline(
        chunking_config=ChunkingConfig(
            default=ChunkingProfileConfig(target_tokens=50, max_tokens=80),
            by_extension={
                ".md": ChunkingProfileConfig(target_tokens=25, max_tokens=40),
                ".txt": ChunkingProfileConfig(target_tokens=10, max_tokens=20),
            },
        )
    )

    report = pipeline.run(raw_dir, store)

    chunks = store.load()
    chunks_by_extension = {chunk.metadata["extension"]: chunk for chunk in chunks}
    assert report.documents_new == 2
    assert chunks_by_extension[".md"].metadata["chunk_target_tokens"] == "25"
    assert chunks_by_extension[".md"].metadata["chunk_max_tokens"] == "40"
    assert chunks_by_extension[".txt"].metadata["chunk_target_tokens"] == "10"
    assert chunks_by_extension[".txt"].metadata["chunk_max_tokens"] == "20"


def test_incremental_ingest_preserves_csv_as_table_chunk(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("incidents.csv").write_text(
        "service,incident,severity\nAuth Service,AUTH-429,high\nBilling,BILL-204,medium\n",
        encoding="utf-8",
    )
    store = JsonChunkStore(tmp_path / "chunks.json")

    report = IncrementalIngestPipeline().run(raw_dir, store)

    chunks = store.load()
    assert report.documents_new == 1
    assert len(chunks) == 1
    assert chunks[0].metadata["extension"] == ".csv"
    assert chunks[0].metadata["source_format"] == "csv"
    assert "| Auth Service | AUTH-429 | high |" in chunks[0].text
    assert chunks[0].metadata["chunking_strategy"] == "structure_aware"


def test_incremental_ingest_reprocesses_when_chunking_config_changes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    store = JsonChunkStore(tmp_path / "chunks.json")

    first = IncrementalIngestPipeline(
        chunking_config=ChunkingConfig(
            default=ChunkingProfileConfig(target_tokens=25, max_tokens=40),
        )
    ).run(raw_dir, store)
    second = IncrementalIngestPipeline(
        chunking_config=ChunkingConfig(
            default=ChunkingProfileConfig(target_tokens=10, max_tokens=20),
        )
    ).run(raw_dir, store)

    chunks = store.load()
    assert first.documents_new == 1
    assert second.documents_updated == 1
    assert second.documents_unchanged == 0
    assert chunks[0].metadata["chunk_target_tokens"] == "10"
    assert chunks[0].metadata["chunk_max_tokens"] == "20"


def test_cleaner_filters_low_quality_documents() -> None:
    document = Document(id="tiny", source_path="tiny.md", text="short")

    cleaned = DirtyDataCleaner().clean(document)

    assert cleaned is None


def test_cleaner_removes_repeated_lines() -> None:
    document = Document(
        id="doc1",
        source_path="manual.md",
        text=(
            "Confidential\n"
            "Confidential\n"
            "Confidential\n"
            "Confidential\n"
            "Hybrid retrieval combines BM25 and vector search.\n"
        ),
    )

    cleaned = DirtyDataCleaner().clean(document)

    assert cleaned is not None
    assert "Hybrid retrieval combines BM25" in cleaned.text
    assert "Confidential" not in cleaned.text
    assert cleaned.metadata["cleaned"] == "true"


def test_redactor_masks_sensitive_values_before_indexing() -> None:
    document = Document(
        id="doc1",
        source_path="incident.md",
        text=(
            "Contact alice@example.com or 415-555-0100. "
            "SSN 123-45-6789. "
            "Use api_key=abc123456789 and sk-testsecret123456."
        ),
    )

    redacted = SensitiveDataRedactor().redact(document)

    assert "alice@example.com" not in redacted.text
    assert "415-555-0100" not in redacted.text
    assert "123-45-6789" not in redacted.text
    assert "abc123456789" not in redacted.text
    assert "sk-testsecret123456" not in redacted.text
    assert "[REDACTED_EMAIL]" in redacted.text
    assert redacted.metadata["redacted"] == "true"
    assert "email" in redacted.metadata["redaction_types"]


def test_parser_preserves_heading_path_and_tables() -> None:
    document = Document(
        id="doc1",
        source_path="manual.md",
        metadata={"extension": ".md", "doc_type": "manual"},
        text=(
            "# Product Manual\n\n"
            "## Retrieval\n\n"
            "Hybrid retrieval combines BM25 and vector search.\n\n"
            "| Feature | Purpose |\n"
            "| --- | --- |\n"
            "| BM25 | Exact keyword recall |\n"
        ),
    )

    blocks = StructureParser().parse(document)

    assert [block.block_type for block in blocks] == [
        BlockType.TITLE,
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.TABLE,
    ]
    assert blocks[2].heading_path == ("Product Manual", "Retrieval")
    assert blocks[2].metadata["extension"] == ".md"
    assert blocks[2].metadata["doc_type"] == "manual"
    assert blocks[3].block_type == BlockType.TABLE
    assert "Exact keyword recall" in blocks[3].text


def test_chunker_keeps_table_as_single_chunk() -> None:
    document = Document(
        id="doc1",
        source_path="manual.md",
        metadata={"extension": ".md"},
        text=(
            "# Product Manual\n\n"
            "## Retrieval\n\n"
            "Hybrid retrieval combines BM25 and vector search.\n\n"
            "| Feature | Purpose |\n"
            "| --- | --- |\n"
            "| BM25 | Exact keyword recall |\n"
        ),
    )
    blocks = StructureParser().parse(document)

    chunks = StructureAwareChunker(target_tokens=20, max_tokens=40).chunk(blocks)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ("Product Manual", "Retrieval")
    assert "Product Manual > Retrieval" in chunks[0].text
    assert chunks[0].metadata["extension"] == ".md"
    assert "| Feature | Purpose |" in chunks[1].text
    assert chunks[1].source_blocks


def test_ingest_chunks_can_round_trip_through_json_store(tmp_path: Path) -> None:
    document = Document(
        id="doc1",
        source_path="manual.md",
        text="# Manual\n\n## Reranking\n\nReranking improves precision after broad recall.",
    )
    blocks = StructureParser().parse(document)
    chunks = StructureAwareChunker().chunk(blocks)
    store = JsonChunkStore(tmp_path / "chunks.json")

    store.save(chunks)
    loaded = store.load()

    assert loaded == chunks
    assert loaded[0].metadata["source_path"] == "manual.md"


def test_incremental_ingest_reuses_unchanged_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")
    pipeline = IncrementalIngestPipeline()

    first_report = pipeline.run(raw_dir, store)
    first_chunks = store.load()
    second_report = pipeline.run(raw_dir, store)
    second_chunks = store.load()

    assert first_report.documents_new == 1
    assert first_report.documents_unchanged == 0
    assert second_report.documents_new == 0
    assert second_report.documents_unchanged == 1
    assert second_report.chunks_upserted == ()
    assert second_report.chunks_deleted == ()
    assert second_chunks == first_chunks


def test_incremental_ingest_dry_run_does_not_write_index(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    index_path = tmp_path / "chunks.json"
    store = JsonChunkStore(index_path)

    report = IncrementalIngestPipeline().run(raw_dir, store, dry_run=True)

    assert report.dry_run is True
    assert report.documents_new == 1
    assert len(report.chunks_upserted) == 1
    assert not index_path.exists()


def test_incremental_ingest_replaces_changed_document_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")
    pipeline = IncrementalIngestPipeline()

    pipeline.run(raw_dir, store)
    source.write_text("# Guide\n\nHybrid retrieval uses BM25, vectors, and reranking.", encoding="utf-8")
    report = pipeline.run(raw_dir, store)
    chunks = store.load()

    assert report.documents_updated == 1
    assert report.documents_unchanged == 0
    assert len(report.chunks_upserted) == 1
    assert len(report.chunks_deleted) == 1
    assert report.chunks_upserted != report.chunks_deleted
    assert len(chunks) == 1
    assert "reranking" in chunks[0].text


def test_incremental_ingest_removes_chunks_for_deleted_documents(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    keep = raw_dir / "keep.md"
    remove = raw_dir / "remove.md"
    keep.write_text("# Keep\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    remove.write_text("# Remove\n\nLegacy document that should leave the index.", encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")
    pipeline = IncrementalIngestPipeline()

    pipeline.run(raw_dir, store)
    remove.unlink()
    report = pipeline.run(raw_dir, store)
    chunks = store.load()

    assert report.documents_deleted == 1
    assert report.documents_unchanged == 1
    assert len(report.chunks_deleted) == 1
    assert report.chunks_upserted == ()
    assert len(chunks) == 1
    assert chunks[0].metadata["filename"] == "keep.md"


def test_incremental_ingest_keeps_same_source_path_separate_by_tenant(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    store = JsonChunkStore(tmp_path / "chunks.json")
    pipeline = IncrementalIngestPipeline()

    pipeline.run(raw_dir, store, metadata_overrides={"tenant_id": "acme"})
    report = pipeline.run(raw_dir, store, metadata_overrides={"tenant_id": "globex"})
    chunks = store.load()

    assert report.documents_new == 1
    assert len(chunks) == 2
    assert {chunk.metadata["tenant_id"] for chunk in chunks} == {"acme", "globex"}


def test_incremental_ingest_redacts_sensitive_values_in_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "incident.md"
    source.write_text(
        "# Incident\n\nContact alice@example.com with api_key=abc123456789 for hybrid retrieval review.",
        encoding="utf-8",
    )
    store = JsonChunkStore(tmp_path / "chunks.json")

    report = IncrementalIngestPipeline().run(raw_dir, store)
    chunks = store.load()

    assert report.documents_new == 1
    assert "alice@example.com" not in chunks[0].text
    assert "abc123456789" not in chunks[0].text
    assert "[REDACTED_EMAIL]" in chunks[0].text
    assert chunks[0].metadata["redacted"] == "true"
    assert "api_token" in chunks[0].metadata["redaction_types"]
