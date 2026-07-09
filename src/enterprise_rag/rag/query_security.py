from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuerySecurityFinding:
    label: str
    message: str


@dataclass(frozen=True)
class QuerySecurityResult:
    allowed: bool
    findings: tuple[QuerySecurityFinding, ...] = ()


class QueryGuard:
    def __init__(self, max_query_chars: int = 2000, rules: dict[str, tuple[str, str]] | None = None) -> None:
        self.max_query_chars = max_query_chars
        self.rules = {
            "instruction_override": (
                r"\b(ignore|bypass|override).{0,40}\b(instructions?|system prompt|developer message|rules?)\b",
                "Query appears to ask the system to ignore or override instructions.",
            ),
            "secret_exfiltration": (
                r"\b(reveal|print|show|dump|exfiltrate|leak).{0,50}\b(secret|api key|token|password|system prompt)\b",
                "Query appears to request secret or system information.",
            ),
            "bulk_data_dump": (
                r"\b(dump|export|show|list|return).{0,40}\b(all|every).{0,40}\b(document|file|record|contract|customer)\b",
                "Query appears to request a broad data dump.",
            ),
        }
        if rules:
            self.rules.update(rules)

    def check(self, query: str) -> QuerySecurityResult:
        findings = []
        normalized = " ".join(query.split())
        if len(normalized) > self.max_query_chars:
            findings.append(
                QuerySecurityFinding(
                    label="query_too_long",
                    message=f"Query exceeds maximum length of {self.max_query_chars} characters.",
                )
            )

        for label, (pattern, message) in self.rules.items():
            if re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
                findings.append(QuerySecurityFinding(label=label, message=message))

        return QuerySecurityResult(allowed=not findings, findings=tuple(findings))
