from enterprise_rag.evaluation.index_inspection import format_index_quality_report, inspect_index
from enterprise_rag.models import Chunk


def test_inspect_index_reports_chunk_quality_stats() -> None:
    chunks = [
        Chunk(
            id="one",
            document_id="doc1",
            text="Hybrid retrieval combines BM25 and vector search.",
            metadata={"source_path": "one.md", "extension": ".md", "allowed_groups": "security"},
        ),
        Chunk(
            id="two",
            document_id="doc1",
            text="",
            metadata={"extension": ".md"},
        ),
    ]

    report = inspect_index(chunks)

    assert report.chunk_count == 2
    assert report.empty_chunks == 1
    assert report.chunks_missing_source == 1
    assert report.chunks_missing_extension == 0
    assert report.acl_protected_chunks == 1
    assert report.metadata_keys == ("allowed_groups", "extension", "source_path")


def test_format_index_quality_report_is_readable() -> None:
    report = inspect_index(
        [
            Chunk(
                id="one",
                document_id="doc1",
                text="Hybrid retrieval.",
                metadata={"source_path": "one.md", "extension": ".md"},
            )
        ]
    )

    formatted = format_index_quality_report(report)

    assert "Index Quality Report" in formatted
    assert "chunks: 1" in formatted
    assert "metadata_keys: extension, source_path" in formatted

