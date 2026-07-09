from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.models import Document


@dataclass(frozen=True)
class RedactionRule:
    label: str
    pattern: re.Pattern[str]
    replacement: str


class SensitiveDataRedactor:
    def __init__(self, rules: tuple[RedactionRule, ...] | None = None) -> None:
        self.rules = rules or DEFAULT_REDACTION_RULES

    def redact(self, document: Document) -> Document:
        text = document.text
        counts: dict[str, int] = {}
        for rule in self.rules:
            text, count = rule.pattern.subn(rule.replacement, text)
            if count:
                counts[rule.label] = counts.get(rule.label, 0) + count
        if not counts:
            return document
        return Document(
            id=document.id,
            source_path=document.source_path,
            text=text,
            metadata={
                **document.metadata,
                "redacted": "true",
                "redaction_types": ",".join(sorted(counts)),
                "redaction_count": str(sum(counts.values())),
            },
        )


DEFAULT_REDACTION_RULES = (
    RedactionRule(
        label="email",
        pattern=re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE),
        replacement="[REDACTED_EMAIL]",
    ),
    RedactionRule(
        label="ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        replacement="[REDACTED_SSN]",
    ),
    RedactionRule(
        label="phone",
        pattern=re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        replacement="[REDACTED_PHONE]",
    ),
    RedactionRule(
        label="credit_card",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        replacement="[REDACTED_CREDIT_CARD]",
    ),
    RedactionRule(
        label="openai_key",
        pattern=re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        replacement="[REDACTED_SECRET]",
    ),
    RedactionRule(
        label="api_token",
        pattern=re.compile(
            r"\b(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}['\"]?",
            flags=re.IGNORECASE,
        ),
        replacement=r"\1=[REDACTED_SECRET]",
    ),
)
