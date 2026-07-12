from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    request_id: str
    query: str
    answer: str
    rating: str
    tenant_id: str | None = None
    user_id: str | None = None
    citation_chunk_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    comment: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["citation_chunk_ids"] = list(self.citation_chunk_ids)
        payload["labels"] = list(self.labels)
        return payload


class JsonFeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: FeedbackRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def load(self) -> list[FeedbackRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                FeedbackRecord(
                    feedback_id=str(item["feedback_id"]),
                    request_id=str(item["request_id"]),
                    query=str(item["query"]),
                    answer=str(item["answer"]),
                    rating=str(item["rating"]),
                    tenant_id=item.get("tenant_id"),
                    user_id=item.get("user_id"),
                    citation_chunk_ids=tuple(str(value) for value in item.get("citation_chunk_ids", [])),
                    labels=tuple(str(value) for value in item.get("labels", [])),
                    comment=str(item.get("comment", "")),
                    timestamp=float(item.get("timestamp", 0.0)),
                )
            )
        return records
