from pathlib import Path

from enterprise_rag.models import Chunk
from enterprise_rag.storage.index_version import JsonIndexVersionStore, fallback_index_version
from enterprise_rag.storage.json_store import JsonChunkStore


def test_json_index_version_store_bumps_sequence(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.json"
    version_store = JsonIndexVersionStore(tmp_path / "index_version.json")
    index_path.write_text("[]", encoding="utf-8")

    first = version_store.bump(reason="ingest", index_path=index_path)
    second = version_store.bump(reason="ingest", index_path=index_path)

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.version_id != first.version_id
    assert version_store.current() == second
    assert version_store.current_id(index_path) == second.version_id
    assert version_store.history() == (first, second)
    assert Path(first.snapshot_path).exists()
    assert Path(second.snapshot_path).exists()


def test_json_index_version_store_falls_back_to_index_file_version(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.json"
    version_store = JsonIndexVersionStore(tmp_path / "index_version.json")
    index_path.write_text("[]", encoding="utf-8")

    assert version_store.current_id(index_path) == fallback_index_version(index_path)


def test_json_index_version_store_rolls_back_to_snapshot(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.json"
    chunk_store = JsonChunkStore(index_path)
    version_store = JsonIndexVersionStore(tmp_path / "index_version.json")

    chunk_store.save([Chunk(id="old", document_id="doc1", text="Old approved index.")])
    old_version = version_store.bump(reason="ingest", index_path=index_path)
    chunk_store.save([Chunk(id="new", document_id="doc1", text="Bad new index.")])
    new_version = version_store.bump(reason="ingest", index_path=index_path)

    rollback_version = version_store.rollback(version_id=old_version.version_id, index_path=index_path)
    chunks = chunk_store.load()

    assert chunks[0].id == "old"
    assert rollback_version.sequence == 3
    assert rollback_version.version_id not in {old_version.version_id, new_version.version_id}
    assert rollback_version.reason == f"rollback:{old_version.version_id}"
    assert version_store.current() == rollback_version
