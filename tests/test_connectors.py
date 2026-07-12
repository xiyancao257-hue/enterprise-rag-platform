from pathlib import Path

from enterprise_rag.ingestion.connectors import LocalFileConnector, S3LikeConnector
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


def test_s3_like_connector_adds_cloud_source_metadata_and_acl(tmp_path: Path) -> None:
    source_root = tmp_path / "objects"
    source_root.mkdir()
    source_root.joinpath("guide.md").write_text(
        "# Guide\n\nHybrid retrieval combines BM25 and vector search.",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "objects": [
            {
              "key": "docs/guide.md",
              "path": "guide.md",
              "etag": "etag-1",
              "version_id": "version-1",
              "last_modified": "2026-07-12T00:00:00Z",
              "allowed_groups": ["engineering", "support"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    result = S3LikeConnector(bucket="enterprise-docs", manifest_path=manifest, page_size=1).load(source_root)

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.metadata["source_system"] == "s3"
    assert document.metadata["source_uri"] == "s3://enterprise-docs/docs/guide.md"
    assert document.metadata["source_bucket"] == "enterprise-docs"
    assert document.metadata["source_key"] == "docs/guide.md"
    assert document.metadata["source_version"] == "version-1"
    assert document.metadata["source_updated_at"] == "2026-07-12T00:00:00Z"
    assert document.metadata["source_etag"] == "etag-1"
    assert document.metadata["allowed_groups"] == "engineering,support"


def test_s3_like_connector_preserves_filter_report(tmp_path: Path) -> None:
    source_root = tmp_path / "objects"
    source_root.mkdir()
    source_root.joinpath("image.png").write_bytes(b"image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"key": "images/image.png", "path": "image.png"}]', encoding="utf-8")

    result = S3LikeConnector(
        bucket="enterprise-docs",
        manifest_path=manifest,
        policy=IngestionFilePolicy(allowed_extensions=(".md",)),
    ).load(source_root)

    assert result.documents == ()
    assert result.documents_filtered == 1
    assert result.filtered_documents[0].source_path == str(source_root / "image.png")
