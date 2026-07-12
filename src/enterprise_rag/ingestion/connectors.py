from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from enterprise_rag.ingestion.loaders import FilteredDocument, LoadDocumentsResult, load_documents_with_report
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


class S3LikeConnector:
    source_system = "s3"

    def __init__(
        self,
        bucket: str,
        manifest_path: Path,
        policy: IngestionFilePolicy | None = None,
        ocr_adapter: OcrAdapter | None = None,
        pdf_page_renderer: PdfPageRenderer | None = None,
        page_size: int = 100,
    ) -> None:
        self.bucket = bucket
        self.manifest_path = manifest_path
        self.policy = policy or IngestionFilePolicy()
        self.ocr_adapter = ocr_adapter
        self.pdf_page_renderer = pdf_page_renderer
        self.page_size = page_size

    def load(self, source_path: Path) -> LoadDocumentsResult:
        documents: list[Document] = []
        filtered_documents: list[FilteredDocument] = []
        filter_reasons: dict[str, int] = {}
        for page in self._list_pages():
            for entry in page:
                object_path = source_path / str(entry["path"])
                result = load_documents_with_report(
                    object_path,
                    policy=self.policy,
                    ocr_adapter=self.ocr_adapter,
                    pdf_page_renderer=self.pdf_page_renderer,
                )
                documents.extend(self._with_s3_metadata(document, entry) for document in result.documents)
                filtered_documents.extend(result.filtered_documents)
                for reason, count in result.filter_reasons.items():
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + count
        return LoadDocumentsResult(
            documents=tuple(documents),
            documents_filtered=sum(filter_reasons.values()),
            filter_reasons=filter_reasons,
            filtered_documents=tuple(filtered_documents),
        )

    def _list_pages(self) -> tuple[tuple[dict[str, object], ...], ...]:
        entries = _load_manifest_entries(self.manifest_path)
        pages = []
        for start in range(0, len(entries), self.page_size):
            pages.append(tuple(entries[start : start + self.page_size]))
        return tuple(pages)

    def _with_s3_metadata(self, document: Document, entry: dict[str, object]) -> Document:
        key = str(entry["key"])
        etag = str(entry.get("etag", ""))
        version_id = str(entry.get("version_id", etag))
        last_modified = str(entry.get("last_modified", ""))
        allowed_groups = ",".join(str(group) for group in entry.get("allowed_groups", ()))
        metadata = {
            **document.metadata,
            "source_system": self.source_system,
            "source_uri": f"s3://{self.bucket}/{key}",
            "source_bucket": self.bucket,
            "source_key": key,
            "source_version": version_id,
            "source_updated_at": last_modified,
            "source_etag": etag,
        }
        if allowed_groups:
            metadata["allowed_groups"] = allowed_groups
        return replace(
            document,
            id=hashlib.sha256(f"s3://{self.bucket}/{key}".encode()).hexdigest()[:16],
            metadata=metadata,
        )


def _file_uri(path: Path) -> str:
    return path.resolve(strict=False).as_uri()


def _mtime_ns(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return ""


def _load_manifest_entries(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("objects") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("S3-like manifest must be a list or an object with an `objects` list.")
    loaded = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Each S3-like manifest object must be a JSON object.")
        if "key" not in item or "path" not in item:
            raise ValueError("Each S3-like manifest object must include `key` and `path`.")
        loaded.append(item)
    return loaded
