from pathlib import Path

from enterprise_rag.ingestion.sync_manifest import JsonSourceSyncManifestStore
from enterprise_rag.models import Document


def test_json_source_sync_manifest_store_records_active_documents(tmp_path: Path) -> None:
    store = JsonSourceSyncManifestStore(tmp_path / "source_manifest.json")
    document = Document(
        id="doc1",
        source_path="/tmp/guide.md",
        text="Hybrid retrieval combines BM25 and vector search.",
        metadata={
            "source_system": "local_file",
            "source_uri": "file:///tmp/guide.md",
            "source_version": "version-1",
            "source_updated_at": "123",
            "content_hash": "hash-1",
        },
    )

    store.update_from_documents("tenant-a", (document,), seen_at="2026-07-12T00:00:00+00:00")

    entries = store.load()
    assert len(entries) == 1
    assert entries[0].tenant_id == "tenant-a"
    assert entries[0].source_uri == "file:///tmp/guide.md"
    assert entries[0].source_system == "local_file"
    assert entries[0].source_version == "version-1"
    assert entries[0].source_updated_at == "123"
    assert entries[0].content_hash == "hash-1"
    assert entries[0].last_seen_at == "2026-07-12T00:00:00+00:00"
    assert entries[0].status == "active"


def test_json_source_sync_manifest_store_marks_deleted_sources(tmp_path: Path) -> None:
    store = JsonSourceSyncManifestStore(tmp_path / "source_manifest.json")
    document = Document(
        id="doc1",
        source_path="/tmp/guide.md",
        text="Hybrid retrieval combines BM25 and vector search.",
        metadata={
            "source_uri": "file:///tmp/guide.md",
            "content_hash": "hash-1",
        },
    )

    store.update_from_documents("tenant-a", (document,), seen_at="first")
    store.update_from_documents("tenant-a", (), deleted_source_uris=("file:///tmp/guide.md",), seen_at="second")

    entries = store.load()
    assert len(entries) == 1
    assert entries[0].source_uri == "file:///tmp/guide.md"
    assert entries[0].status == "deleted"
    assert entries[0].last_seen_at == "second"
