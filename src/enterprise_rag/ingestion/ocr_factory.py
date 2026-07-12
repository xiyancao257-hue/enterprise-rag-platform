from __future__ import annotations

from enterprise_rag.config import OcrConfig
from enterprise_rag.ingestion.ocr import (
    DisabledOcrAdapter,
    OcrAdapter,
    PdfPageRenderer,
    PopplerPdfPageRenderer,
    TesseractOcrAdapter,
    UnconfiguredOcrAdapter,
)


def create_ocr_adapter(config: OcrConfig) -> OcrAdapter:
    provider = config.provider.lower().strip()
    if provider in {"disabled", "none", ""}:
        return DisabledOcrAdapter()
    if provider == "tesseract":
        return TesseractOcrAdapter(
            command=config.tesseract_cmd,
            timeout_seconds=config.tesseract_timeout_seconds,
        )
    if provider == "aws_textract":
        region_hint = f" in region `{config.aws_region}`" if config.aws_region else ""
        return UnconfiguredOcrAdapter(
            provider=provider,
            setup_hint=f"Configure AWS Textract credentials{region_hint} and wire a boto3 adapter.",
        )
    if provider == "azure_document_intelligence":
        return UnconfiguredOcrAdapter(
            provider=provider,
            setup_hint=(
                "Configure Azure Document Intelligence credentials from "
                f"`{config.azure_endpoint_env_var}` and `{config.azure_key_env_var}`."
            ),
        )
    raise ValueError(f"Unsupported OCR provider `{config.provider}`.")


def create_pdf_page_renderer(config: OcrConfig) -> PdfPageRenderer:
    return PopplerPdfPageRenderer(
        command=config.pdf_renderer_cmd,
        dpi=config.pdf_render_dpi,
        timeout_seconds=config.pdf_render_timeout_seconds,
    )
