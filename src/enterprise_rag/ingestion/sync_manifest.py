from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from enterprise_rag.models import Document


@dataclass(frozen=True)
class SourceManifestEntry:
    tenant_id: str
    source_uri: str
    source_system: str
    source_version: str
    source_updated_at: str
    content_hash: str
    last_seen_at: str
    status: str = "active"


class SourceSyncManifestStore(Protocol):
    def load(self) -> list[SourceManifestEntry]:
        """Load the current source sync manifest."""

    def update_from_documents(
        self,
        tenant_id: str,
        documents: tuple[Document, ...],
        deleted_source_uris: tuple[str, ...] = (),
        seen_at: str | None = None,
    ) -> None:
        """Mark loaded source documents active and missing source URIs deleted."""


class JsonSourceSyncManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[SourceManifestEntry]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            SourceManifestEntry(
                tenant_id=str(item["tenant_id"]),
                source_uri=str(item["source_uri"]),
                source_system=str(item.get("source_system", "")),
                source_version=str(item.get("source_version", "")),
                source_updated_at=str(item.get("source_updated_at", "")),
                content_hash=str(item.get("content_hash", "")),
                last_seen_at=str(item.get("last_seen_at", "")),
                status=str(item.get("status", "active")),
            )
            for item in payload
        ]

    def update_from_documents(
        self,
        tenant_id: str,
        documents: tuple[Document, ...],
        deleted_source_uris: tuple[str, ...] = (),
        seen_at: str | None = None,
    ) -> None:
        seen_at = seen_at or _now_iso()
        entries = {(entry.tenant_id, entry.source_uri): entry for entry in self.load()}

        for document in documents:
            source_uri = document.metadata.get("source_uri") or document.source_path
            entries[(tenant_id, source_uri)] = SourceManifestEntry(
                tenant_id=tenant_id,
                source_uri=source_uri,
                source_system=document.metadata.get("source_system", ""),
                source_version=document.metadata.get("source_version", ""),
                source_updated_at=document.metadata.get("source_updated_at", ""),
                content_hash=document.metadata.get("content_hash", ""),
                last_seen_at=seen_at,
                status="active",
            )

        for source_uri in deleted_source_uris:
            key = (tenant_id, source_uri)
            existing = entries.get(key)
            if existing is None:
                entries[key] = SourceManifestEntry(
                    tenant_id=tenant_id,
                    source_uri=source_uri,
                    source_system="",
                    source_version="",
                    source_updated_at="",
                    content_hash="",
                    last_seen_at=seen_at,
                    status="deleted",
                )
                continue
            entries[key] = replace(existing, last_seen_at=seen_at, status="deleted")

        self._save(list(entries.values()))

    def _save(self, entries: list[SourceManifestEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(entries, key=lambda entry: (entry.tenant_id, entry.source_uri))
        self.path.write_text(json.dumps([asdict(entry) for entry in ordered], indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
