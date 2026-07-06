from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from enterprise_rag.models import Chunk


class JsonChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, chunks: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(chunk) for chunk in chunks]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> list[Chunk]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            Chunk(
                id=item["id"],
                document_id=item["document_id"],
                text=item["text"],
                heading_path=tuple(item.get("heading_path", [])),
                source_blocks=tuple(item.get("source_blocks", [])),
                metadata=item.get("metadata", {}),
            )
            for item in payload
        ]

