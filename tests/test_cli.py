from pathlib import Path

from enterprise_rag import cli


class RecordingVectorSync:
    calls = []

    def sync(self, vector_index: object, chunks_to_upsert: list[object], chunk_ids_to_delete: list[str]) -> object:
        self.calls.append((vector_index, chunks_to_upsert, chunk_ids_to_delete))
        return type(
            "SyncReport",
            (),
            {"vectors_upserted": len(chunks_to_upsert), "vectors_deleted": len(chunk_ids_to_delete)},
        )()


def test_ingest_cli_can_sync_changed_chunks_to_vector_index(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "guide.md"
    source.write_text("# Guide\n\nHybrid retrieval combines BM25 and vector search.", encoding="utf-8")
    index_path = tmp_path / "chunks.json"
    fake_vector_index = object()
    RecordingVectorSync.calls.clear()
    monkeypatch.setattr(cli, "create_vector_index", lambda config: fake_vector_index)
    monkeypatch.setattr(cli, "VectorIndexSync", RecordingVectorSync)

    cli.ingest(raw_dir, index_path, sync_vectors=True)

    output = capsys.readouterr().out
    assert "Vector sync report: upserted=1, deleted=0" in output
    assert RecordingVectorSync.calls[0][0] is fake_vector_index
    assert len(RecordingVectorSync.calls[0][1]) == 1
    assert RecordingVectorSync.calls[0][2] == []
