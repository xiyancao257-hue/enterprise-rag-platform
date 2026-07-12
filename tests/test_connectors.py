from pathlib import Path

from enterprise_rag.ingestion.connectors import LocalFileConnector
from enterprise_rag.ingestion.policy import IngestionFilePolicy


def test_local_file_connector_adds_source_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")

    result = LocalFileConnector().load(raw_dir)

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.metadata["source_system"] == "local_file"
    assert document.metadata["source_uri"] == source.resolve().as_uri()
    assert document.metadata["source_version"] == document.metadata["content_hash"]
    assert document.metadata["source_updated_at"]


def test_local_file_connector_keeps_filter_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("unsupported.png").write_bytes(b"image")

    result = LocalFileConnector(policy=IngestionFilePolicy(allowed_extensions=(".md",))).load(raw_dir)

    assert result.documents == ()
    assert result.documents_filtered == 1
    assert result.filtered_documents[0].source_path == str(raw_dir / "unsupported.png")
