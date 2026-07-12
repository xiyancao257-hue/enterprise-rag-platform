from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.config import IngestionConfig

FILTER_UNSUPPORTED_EXTENSION = "unsupported_extension"
FILTER_FILE_TOO_LARGE = "file_too_large"


@dataclass(frozen=True)
class IngestionFilePolicy:
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".csv")
    max_file_bytes: int = 10_000_000

    @classmethod
    def from_config(cls, config: IngestionConfig) -> IngestionFilePolicy:
        return cls(
            allowed_extensions=tuple(extension.lower() for extension in config.allowed_extensions),
            max_file_bytes=config.max_file_bytes,
        )

    def allows(self, path: Path) -> bool:
        return self.rejection_reason(path) is None

    def rejection_reason(self, path: Path) -> str | None:
        extension = path.suffix.lower()
        if extension not in self.allowed_extensions:
            return FILTER_UNSUPPORTED_EXTENSION
        if self.max_file_bytes > 0 and path.stat().st_size > self.max_file_bytes:
            return FILTER_FILE_TOO_LARGE
        return None
