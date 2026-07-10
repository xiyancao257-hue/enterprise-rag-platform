from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.models import Document
from enterprise_rag.text import normalize_text


@dataclass(frozen=True)
class LoadDocumentsResult:
    documents: tuple[Document, ...]
    documents_filtered: int


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
    documents_filtered = 0
    for file in files:
        if not policy.allows(file):
            documents_filtered += 1
            continue
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw_text)
        if not text:
            documents_filtered += 1
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
    return LoadDocumentsResult(documents=tuple(documents), documents_filtered=documents_filtered)
