from pathlib import Path

from enterprise_rag.ingestion.loaders import load_documents
from enterprise_rag.ingestion.pipeline import IncrementalIngestPipeline
from enterprise_rag.models import BlockType, Document
from enterprise_rag.processing.chunking import StructureAwareChunker
from enterprise_rag.processing.cleaning import DirtyDataCleaner
from enterprise_rag.processing.parser import StructureParser
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
