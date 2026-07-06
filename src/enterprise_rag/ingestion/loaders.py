from __future__ import annotations

import hashlib
from pathlib import Path

from enterprise_rag.models import Document
from enterprise_rag.text import normalize_text

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(path: Path) -> list[Document]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(file for file in path.rglob("*") if file.suffix.lower() in SUPPORTED_EXTENSIONS)

    documents: list[Document] = []
    for file in files:
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw_text)
        if not text:
            continue
        doc_id = hashlib.sha256(str(file.resolve()).encode("utf-8")).hexdigest()[:16]
        documents.append(
            Document(
                id=doc_id,
                source_path=str(file),
                text=text,
                metadata={"extension": file.suffix.lower(), "filename": file.name},
            )
        )
    return documents
