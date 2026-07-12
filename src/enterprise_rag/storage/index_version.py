from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class IndexVersion:
    version_id: str
    sequence: int
    updated_at: str
    reason: str


class JsonIndexVersionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def current(self) -> IndexVersion | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return IndexVersion(
            version_id=str(payload["version_id"]),
            sequence=int(payload.get("sequence", 0)),
            updated_at=str(payload.get("updated_at", "")),
            reason=str(payload.get("reason", "")),
        )

    def current_id(self, fallback_index_path: Path | None = None) -> str:
        version = self.current()
        if version is not None:
            return version.version_id
        if fallback_index_path is None:
            return "missing"
        return fallback_index_version(fallback_index_path)

    def bump(self, *, reason: str, index_path: Path | None = None) -> IndexVersion:
        previous = self.current()
        sequence = 1 if previous is None else previous.sequence + 1
        updated_at = _now_iso()
        fingerprint = {
            "sequence": sequence,
            "updated_at": updated_at,
            "reason": reason,
            "fallback_index_version": fallback_index_version(index_path) if index_path is not None else "",
        }
        encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        version = IndexVersion(
            version_id=hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
            sequence=sequence,
            updated_at=updated_at,
            reason=reason,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(version), indent=2), encoding="utf-8")
        return version


def fallback_index_version(index_path: Path) -> str:
    if not index_path.exists():
        return "missing"
    stat = index_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
