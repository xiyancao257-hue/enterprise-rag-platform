from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class OcrUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrResult:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class OcrAdapter(Protocol):
    def extract_text(self, path: Path) -> OcrResult:
        """Extract text from an image or scanned document."""


class DisabledOcrAdapter:
    def extract_text(self, path: Path) -> OcrResult:
        raise OcrUnavailableError(f"OCR is not configured for {path}.")
