from __future__ import annotations

from collections import Counter

from enterprise_rag.models import Document
from enterprise_rag.text import normalize_text


class DirtyDataCleaner:
    def clean(self, document: Document) -> Document | None:
        text = normalize_text(document.text)
        if self._is_low_quality(text):
            return None
        text = self._remove_repeated_lines(text)
        return Document(
            id=document.id,
            source_path=document.source_path,
            text=text,
            metadata={**document.metadata, "cleaned": "true"},
        )

    def _is_low_quality(self, text: str) -> bool:
        if len(text) < 20:
            return True
        visible = sum(1 for char in text if char.isprintable() and not char.isspace())
        if visible / max(len(text), 1) < 0.35:
            return True
        alpha_num = sum(1 for char in text if char.isalnum())
        return alpha_num / max(len(text), 1) < 0.25

    def _remove_repeated_lines(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        counts = Counter(line for line in lines if line)
        cleaned = [line for line in lines if not line or counts[line] <= 3]
        return normalize_text("\n".join(cleaned))

