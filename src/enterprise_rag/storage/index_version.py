from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class IndexVersion:
    version_id: str
    sequence: int
    updated_at: str
    reason: str
    snapshot_path: str = ""


class JsonIndexVersionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def current(self) -> IndexVersion | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return _version_from_payload(_current_payload(payload))

    def history(self) -> tuple[IndexVersion, ...]:
        if not self.path.exists():
            return ()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if "history" not in payload:
            return (_version_from_payload(payload),)
        return tuple(_version_from_payload(item) for item in payload.get("history", []))

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
        version = self._build_version(sequence=sequence, reason=reason, index_path=index_path)
        if index_path is not None and index_path.exists():
            snapshot_path = self._snapshot_path(version.version_id)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(index_path, snapshot_path)
            version = IndexVersion(**{**asdict(version), "snapshot_path": str(snapshot_path)})
        self._write_current(version)
        return version

    def rollback(self, *, version_id: str, index_path: Path, reason: str = "rollback") -> IndexVersion:
        target = self.get(version_id)
        if target is None:
            raise ValueError(f"Index version `{version_id}` was not found.")
        if not target.snapshot_path:
            raise ValueError(f"Index version `{version_id}` does not have a rollback snapshot.")
        snapshot_path = Path(target.snapshot_path)
        if not snapshot_path.exists():
            raise ValueError(f"Rollback snapshot for index version `{version_id}` is missing.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot_path, index_path)

        previous = self.current()
        sequence = 1 if previous is None else previous.sequence + 1
        version = self._build_version(
            sequence=sequence,
            reason=f"{reason}:{version_id}",
            index_path=index_path,
        )
        rollback_snapshot_path = self._snapshot_path(version.version_id)
        rollback_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(index_path, rollback_snapshot_path)
        version = IndexVersion(**{**asdict(version), "snapshot_path": str(rollback_snapshot_path)})
        self._write_current(version)
        return version

    def get(self, version_id: str) -> IndexVersion | None:
        for version in self.history():
            if version.version_id == version_id:
                return version
        return None

    def _build_version(self, *, sequence: int, reason: str, index_path: Path | None = None) -> IndexVersion:
        updated_at = _now_iso()
        fingerprint = {
            "sequence": sequence,
            "updated_at": updated_at,
            "reason": reason,
            "fallback_index_version": fallback_index_version(index_path) if index_path is not None else "",
        }
        encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        return IndexVersion(
            version_id=hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
            sequence=sequence,
            updated_at=updated_at,
            reason=reason,
        )

    def _write_current(self, version: IndexVersion) -> None:
        history = [asdict(item) for item in self.history()]
        history.append(asdict(version))
        payload = {
            "current": asdict(version),
            "history": history,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _snapshot_path(self, version_id: str) -> Path:
        return self.path.with_name("index_versions") / f"{version_id}.json"


def fallback_index_version(index_path: Path) -> str:
    if not index_path.exists():
        return "missing"
    stat = index_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _current_payload(payload: dict[str, object]) -> dict[str, object]:
    current = payload.get("current")
    if isinstance(current, dict):
        return current
    return payload


def _version_from_payload(payload: dict[str, object]) -> IndexVersion:
    return IndexVersion(
        version_id=str(payload["version_id"]),
        sequence=int(payload.get("sequence", 0)),
        updated_at=str(payload.get("updated_at", "")),
        reason=str(payload.get("reason", "")),
        snapshot_path=str(payload.get("snapshot_path", "")),
    )
