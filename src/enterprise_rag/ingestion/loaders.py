from __future__ import annotations

import csv
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
        raw_text = _read_file_text(file)
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
                metadata={
                    "extension": file.suffix.lower(),
                    "filename": file.name,
                    "content_hash": content_hash,
                    **_loader_metadata(file),
                },
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


def _read_file_text(file: Path) -> str:
    if file.suffix.lower() == ".csv":
        return _csv_to_markdown_table(file)
    return file.read_text(encoding="utf-8", errors="ignore")


def _csv_to_markdown_table(file: Path) -> str:
    rows = []
    with file.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            cleaned = [cell.strip().replace("\n", " ") for cell in row]
            if any(cleaned):
                rows.append(cleaned)
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    table_lines = [
        _markdown_row(header),
        _markdown_row(["---"] * width),
        *(_markdown_row(row) for row in body),
    ]
    title = file.stem.replace("_", " ").replace("-", " ").title()
    return "\n".join([f"# {title}", "", *table_lines])


def _markdown_row(cells: list[str]) -> str:
    escaped = [cell.replace("|", "\\|") for cell in cells]
    return "| " + " | ".join(escaped) + " |"


def _loader_metadata(file: Path) -> dict[str, str]:
    if file.suffix.lower() == ".csv":
        return {"source_format": "csv", "table_format": "markdown"}
    return {}
