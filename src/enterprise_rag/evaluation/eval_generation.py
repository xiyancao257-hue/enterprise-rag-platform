from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_rag.observability.log_analysis import load_query_log_records


@dataclass(frozen=True)
class GeneratedEvalCase:
    id: str
    query: str
    expected_text_contains: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_text_contains": list(self.expected_text_contains),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EvalPromotionReport:
    promoted_count: int
    skipped_count: int
    skipped_ids: tuple[str, ...]


def generate_eval_cases_from_logs(path: Path, limit: int = 20) -> list[GeneratedEvalCase]:
    records = load_query_log_records(path)
    candidates = _candidate_records(records)
    cases = []
    seen_queries = set()

    for record in candidates:
        query = str(record.get("query", "")).strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        cases.append(
            GeneratedEvalCase(
                id=f"log_{len(cases) + 1}_{_slugify(query)}",
                query=query,
                expected_text_contains=(),
                notes=_notes_for_record(record),
            )
        )
        if len(cases) >= limit:
            break

    return cases


def write_generated_eval_cases(cases: list[GeneratedEvalCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [case.to_dict() for case in cases]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def promote_reviewed_eval_draft(draft_path: Path, output_path: Path) -> EvalPromotionReport:
    draft_cases = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft_cases, list):
        raise ValueError("Eval draft must contain a JSON array.")

    promoted = []
    skipped_ids = []
    for case in draft_cases:
        if not isinstance(case, dict):
            raise ValueError("Each eval draft case must be a JSON object.")
        if _has_expected_evidence(case):
            promoted.append(_formal_eval_case(case))
        else:
            skipped_ids.append(str(case.get("id", "unknown")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EvalPromotionReport(
        promoted_count=len(promoted),
        skipped_count=len(skipped_ids),
        skipped_ids=tuple(skipped_ids),
    )


def _candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("insufficient_evidence") is True
        or not record.get("retrieved_chunk_ids", [])
        or not record.get("final_chunk_ids", [])
    ]


def _notes_for_record(record: dict[str, Any]) -> str:
    reasons = []
    if record.get("insufficient_evidence") is True:
        reasons.append("insufficient evidence")
    if not record.get("retrieved_chunk_ids", []):
        reasons.append("no retrieval hits")
    if not record.get("final_chunk_ids", []):
        reasons.append("no final context")

    reason_text = ", ".join(reasons) if reasons else "query log review"
    return (
        f"Generated from query log because of {reason_text}. "
        "Fill expected_text_contains after manual review before using this as a regression eval case."
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:40] or "query"


def _has_expected_evidence(case: dict[str, Any]) -> bool:
    return bool(case.get("expected_text_contains")) or bool(case.get("expected_chunk_ids"))


def _formal_eval_case(case: dict[str, Any]) -> dict[str, Any]:
    promoted = {
        "id": case["id"],
        "query": case["query"],
    }
    if case.get("expected_text_contains"):
        promoted["expected_text_contains"] = list(case["expected_text_contains"])
    if case.get("expected_chunk_ids"):
        promoted["expected_chunk_ids"] = list(case["expected_chunk_ids"])
    return promoted
