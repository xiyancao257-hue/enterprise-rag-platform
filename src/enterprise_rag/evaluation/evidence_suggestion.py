from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class SuggestedEvidence:
    chunk_id: str
    source_path: str | None
    heading_path: tuple[str, ...]
    score: float
    retriever: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "heading_path": list(self.heading_path),
            "score": self.score,
            "retriever": self.retriever,
            "text": self.text,
        }


def suggest_evidence_for_eval_draft(
    draft_path: Path,
    chunks: list[Chunk],
    output_path: Path,
    top_k: int = 3,
) -> None:
    draft_cases = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft_cases, list):
        raise ValueError("Eval draft must contain a JSON array.")

    retriever = HybridRetriever(chunks)
    enriched_cases = []
    for case in draft_cases:
        if not isinstance(case, dict):
            raise ValueError("Each eval draft case must be a JSON object.")
        query = str(case.get("query", "")).strip()
        enriched = dict(case)
        enriched["suggested_evidence"] = [
            evidence.to_dict() for evidence in _suggest_for_query(retriever, query, top_k=top_k)
        ]
        enriched_cases.append(enriched)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched_cases, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def approve_suggested_evidence(
    draft_path: Path,
    case_id: str,
    suggestion_index: int,
    output_path: Path,
) -> None:
    draft_cases = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft_cases, list):
        raise ValueError("Eval draft must contain a JSON array.")

    approved = []
    matched = False
    for case in draft_cases:
        if not isinstance(case, dict):
            raise ValueError("Each eval draft case must be a JSON object.")
        updated = dict(case)
        if updated.get("id") == case_id:
            matched = True
            suggestions = updated.get("suggested_evidence", [])
            if not isinstance(suggestions, list):
                raise ValueError(f"Case {case_id} must contain a suggested_evidence array.")
            if suggestion_index < 0 or suggestion_index >= len(suggestions):
                raise IndexError(f"Suggestion index {suggestion_index} is out of range for case {case_id}.")
            suggestion = suggestions[suggestion_index]
            if not isinstance(suggestion, dict) or not suggestion.get("text"):
                raise ValueError(f"Suggestion {suggestion_index} for case {case_id} must contain text.")
            updated["expected_text_contains"] = [suggestion["text"]]
        approved.append(updated)

    if not matched:
        raise ValueError(f"Case id not found: {case_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(approved, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _suggest_for_query(retriever: HybridRetriever, query: str, top_k: int) -> list[SuggestedEvidence]:
    if not query:
        return []
    return [_to_suggestion(hit) for hit in retriever.search([query], top_k=top_k)]


def _to_suggestion(hit: SearchHit) -> SuggestedEvidence:
    return SuggestedEvidence(
        chunk_id=hit.chunk.id,
        source_path=hit.chunk.metadata.get("source_path"),
        heading_path=hit.chunk.heading_path,
        score=hit.score,
        retriever=hit.retriever,
        text=hit.chunk.text,
    )
