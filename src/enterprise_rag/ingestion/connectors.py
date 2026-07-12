from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from enterprise_rag.ingestion.loaders import LoadDocumentsResult, load_documents_with_report
from enterprise_rag.ingestion.ocr import OcrAdapter, PdfPageRenderer
from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.models import Document


class SourceConnector(Protocol):
    def load(self, source_path: Path) -> LoadDocumentsResult:
        """Load source documents and preserve source-system metadata."""


class LocalFileConnector:
    source_system = "local_file"

    def __init__(
        self,
        policy: IngestionFilePolicy | None = None,
        ocr_adapter: OcrAdapter | None = None,
        pdf_page_renderer: PdfPageRenderer | None = None,
    ) -> None:
        self.policy = policy or IngestionFilePolicy()
        self.ocr_adapter = ocr_adapter
        self.pdf_page_renderer = pdf_page_renderer

    def load(self, source_path: Path) -> LoadDocumentsResult:
        result = load_documents_with_report(
            source_path,
            policy=self.policy,
            ocr_adapter=self.ocr_adapter,
            pdf_page_renderer=self.pdf_page_renderer,
        )
        return replace(
            result,
            documents=tuple(self._with_source_metadata(document) for document in result.documents),
        )

    def _with_source_metadata(self, document: Document) -> Document:
        path = Path(document.source_path)
        metadata = {
            **document.metadata,
            "source_system": self.source_system,
            "source_uri": _file_uri(path),
            "source_version": document.metadata.get("content_hash", ""),
            "source_updated_at": _mtime_ns(path),
        }
        return replace(document, metadata=metadata)


def _file_uri(path: Path) -> str:
    return path.resolve(strict=False).as_uri()


def _mtime_ns(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return ""
