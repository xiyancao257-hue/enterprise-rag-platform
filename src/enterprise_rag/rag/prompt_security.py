from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.models import SearchHit


@dataclass(frozen=True)
class PromptInjectionFinding:
    label: str
    pattern: str


@dataclass(frozen=True)
class PromptSecurityResult:
    safe_hits: list[SearchHit]
    blocked_hits: list[SearchHit]
    findings_by_chunk_id: dict[str, tuple[PromptInjectionFinding, ...]]


class PromptInjectionDetector:
    def __init__(self, rules: dict[str, str] | None = None) -> None:
        self.rules = {
            "ignore_instructions": r"\bignore (all )?(previous|prior|above|system|developer) instructions\b",
            "reveal_secrets": (
                r"\b(reveal|print|show|exfiltrate|leak).{0,40}"
                r"\b(secret|api key|token|password|system prompt)\b"
            ),
            "role_override": r"\b(system|developer) (message|prompt|instruction)s?\b",
            "disable_safety": r"\b(disable|bypass|turn off).{0,30}\b(safety|guardrails|policy)\b",
            "citation_bypass": r"\b(do not|don't|never) cite\b",
        }
        if rules:
            self.rules.update(rules)

    def detect(self, text: str) -> tuple[PromptInjectionFinding, ...]:
        findings = []
        for label, pattern in self.rules.items():
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                findings.append(PromptInjectionFinding(label=label, pattern=pattern))
        return tuple(findings)

    def filter_hits(self, hits: list[SearchHit]) -> PromptSecurityResult:
        safe_hits = []
        blocked_hits = []
        findings_by_chunk_id = {}
        for hit in hits:
            findings = self.detect(hit.chunk.text)
            if findings:
                blocked_hits.append(hit)
                findings_by_chunk_id[hit.chunk.id] = findings
            else:
                safe_hits.append(hit)
        return PromptSecurityResult(
            safe_hits=safe_hits,
            blocked_hits=blocked_hits,
            findings_by_chunk_id=findings_by_chunk_id,
        )
