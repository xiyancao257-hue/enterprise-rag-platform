from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.models import Document
from enterprise_rag.text import normalize_text

FILTER_EMPTY_TEXT = "empty_text"


@dataclass(frozen=True)
class FilteredDocument:
    source_path: str
    reason: str


@dataclass(frozen=True)
class LoadDocumentsResult:
    documents: tuple[Document, ...]
    documents_filtered: int
    filter_reasons: dict[str, int]
    filtered_documents: tuple[FilteredDocument, ...] = ()


def load_documents(path: Path) -> list[Document]:
    return list(load_documents_with_report(path).documents)


def load_documents_with_report(
    path: Path,
    policy: IngestionFilePolicy | None = None,
) -> LoadDocumentsResult:
    policy = policy or IngestionFilePolicy()
    if path.is_file():
        files = [path]
    else:
        files = sorted(file for file in path.rglob("*") if file.is_file())

    documents: list[Document] = []
    filter_reasons: dict[str, int] = {}
    filtered_documents: list[FilteredDocument] = []
    for file in files:
        rejection_reason = policy.rejection_reason(file)
        if rejection_reason is not None:
            _count_filter_reason(filter_reasons, rejection_reason)
            filtered_documents.append(FilteredDocument(source_path=str(file), reason=rejection_reason))
            continue
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw_text)
        if not text:
            _count_filter_reason(filter_reasons, FILTER_EMPTY_TEXT)
            filtered_documents.append(FilteredDocument(source_path=str(file), reason=FILTER_EMPTY_TEXT))
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = hashlib.sha256(str(file.resolve()).encode("utf-8")).hexdigest()[:16]
        documents.append(
            Document(
                id=doc_id,
                source_path=str(file),
                text=text,
                metadata={"extension": file.suffix.lower(), "filename": file.name, "content_hash": content_hash},
            )
        )
    return LoadDocumentsResult(
        documents=tuple(documents),
        documents_filtered=sum(filter_reasons.values()),
        filter_reasons=filter_reasons,
        filtered_documents=tuple(filtered_documents),
    )


def _count_filter_reason(filter_reasons: dict[str, int], reason: str) -> None:
    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
