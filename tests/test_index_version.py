from pathlib import Path

from enterprise_rag.storage.index_version import JsonIndexVersionStore, fallback_index_version


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


def test_json_index_version_store_falls_back_to_index_file_version(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.json"
    version_store = JsonIndexVersionStore(tmp_path / "index_version.json")
    index_path.write_text("[]", encoding="utf-8")

    assert version_store.current_id(index_path) == fallback_index_version(index_path)
