from __future__ import annotations

import csv
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.ingestion.ocr import DisabledOcrAdapter, OcrAdapter, OcrUnavailableError, PdfPageRenderer
from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.models import Document
from enterprise_rag.text import normalize_text

FILTER_EMPTY_TEXT = "empty_text"
FILTER_OCR_UNAVAILABLE = "ocr_unavailable"
FILTER_TEXT_EXTRACTION_FAILED = "text_extraction_failed"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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
    ocr_adapter: OcrAdapter | None = None,
    pdf_page_renderer: PdfPageRenderer | None = None,
) -> LoadDocumentsResult:
    policy = policy or IngestionFilePolicy()
    ocr_adapter = ocr_adapter or DisabledOcrAdapter()
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
        try:
            raw_text, loader_metadata = _read_file(file, ocr_adapter, pdf_page_renderer)
        except OcrUnavailableError:
            _count_filter_reason(filter_reasons, FILTER_OCR_UNAVAILABLE)
            filtered_documents.append(FilteredDocument(source_path=str(file), reason=FILTER_OCR_UNAVAILABLE))
            continue
        except Exception:
            _count_filter_reason(filter_reasons, FILTER_TEXT_EXTRACTION_FAILED)
            filtered_documents.append(FilteredDocument(source_path=str(file), reason=FILTER_TEXT_EXTRACTION_FAILED))
            continue
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
                    **loader_metadata,
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


def _read_file(
    file: Path,
    ocr_adapter: OcrAdapter,
    pdf_page_renderer: PdfPageRenderer | None,
) -> tuple[str, dict[str, str]]:
    if file.suffix.lower() == ".csv":
        return _csv_to_markdown_table(file), {"source_format": "csv", "table_format": "markdown"}
    if file.suffix.lower() == ".pdf":
        text, metadata = _pdf_to_text(file)
        if metadata["pdf_pages_with_text"] != "0":
            return text, metadata
        return _ocr_textless_pdf(file, metadata, ocr_adapter, pdf_page_renderer)
    if file.suffix.lower() in IMAGE_EXTENSIONS:
        result = ocr_adapter.extract_text(file)
        return result.text, {**result.metadata, "source_format": "image_ocr"}
    return file.read_text(encoding="utf-8", errors="ignore"), {}


def _ocr_textless_pdf(
    file: Path,
    pdf_metadata: dict[str, str],
    ocr_adapter: OcrAdapter,
    pdf_page_renderer: PdfPageRenderer | None,
) -> tuple[str, dict[str, str]]:
    if pdf_page_renderer is None:
        result = ocr_adapter.extract_text(file)
        return result.text, {**pdf_metadata, **result.metadata, "source_format": "pdf_ocr"}

    title = file.stem.replace("_", " ").replace("-", " ").title()
    parts = [f"# {title}"]
    ocr_metadata: dict[str, str] = {}
    pages_with_ocr_text = 0
    with tempfile.TemporaryDirectory(prefix="enterprise-rag-pdf-ocr-") as temp_dir:
        page_images = pdf_page_renderer.render_pages(file, Path(temp_dir))
        for page_number, page_image in enumerate(page_images, start=1):
            result = ocr_adapter.extract_text(page_image)
            page_text = normalize_text(result.text)
            ocr_metadata.update(result.metadata)
            if not page_text:
                continue
            pages_with_ocr_text += 1
            parts.extend(["", f"## Page {page_number}", "", page_text])

    return "\n".join(parts), {
        **pdf_metadata,
        **ocr_metadata,
        "source_format": "pdf_ocr",
        "pdf_ocr_pages_rendered": str(len(page_images)),
        "pdf_ocr_pages_with_text": str(pages_with_ocr_text),
    }


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


def _pdf_to_text(file: Path) -> tuple[str, dict[str, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(file))
    title = file.stem.replace("_", " ").replace("-", " ").title()
    parts = [f"# {title}"]
    pages_with_text = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        if not page_text:
            continue
        pages_with_text += 1
        parts.extend(["", f"## Page {page_number}", "", page_text])
    return "\n".join(parts), {
        "source_format": "pdf",
        "pdf_page_count": str(len(reader.pages)),
        "pdf_pages_with_text": str(pages_with_text),
    }
