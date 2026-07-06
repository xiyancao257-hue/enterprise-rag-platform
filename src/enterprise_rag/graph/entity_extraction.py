from __future__ import annotations

import re

ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)?\b")
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b")


class RuleBasedEntityExtractor:
    def extract(self, text: str) -> set[str]:
        entities = set()
        entities.update(match.group(0).strip() for match in ACRONYM_RE.finditer(text))
        entities.update(match.group(0).strip() for match in TITLE_CASE_RE.finditer(text))
        return {entity for entity in entities if len(entity) > 1}

    def normalize(self, entity: str) -> str:
        return " ".join(entity.lower().split())

