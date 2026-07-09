from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    request_id: str
    tenant_id: str | None = None
    principal: str | None = None
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, object] = field(default_factory=dict)


class AuditLogger(Protocol):
    def log(self, event: AuditEvent) -> None:
        """Persist one audit event."""


class NullAuditLogger:
    def log(self, event: AuditEvent) -> None:
        return None


class JsonAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, event: AuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event), sort_keys=True) + "\n")
