from pathlib import Path

import pytest

from enterprise_rag.config import OcrConfig
from enterprise_rag.ingestion.ocr import DisabledOcrAdapter, OcrUnavailableError, UnconfiguredOcrAdapter
from enterprise_rag.ingestion.ocr_factory import create_ocr_adapter


def test_disabled_ocr_adapter_raises_clear_error(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"

    with pytest.raises(OcrUnavailableError, match="OCR is not configured"):
        DisabledOcrAdapter().extract_text(image)


def test_ocr_factory_returns_disabled_adapter_by_default() -> None:
    adapter = create_ocr_adapter(OcrConfig())

    assert isinstance(adapter, DisabledOcrAdapter)


@pytest.mark.parametrize(
    ("provider", "expected_hint"),
    [
        ("tesseract", "Install Tesseract"),
        ("aws_textract", "AWS Textract"),
        ("azure_document_intelligence", "Azure Document Intelligence"),
    ],
)
def test_ocr_factory_returns_provider_placeholder(provider: str, expected_hint: str, tmp_path: Path) -> None:
    adapter = create_ocr_adapter(OcrConfig(provider=provider, aws_region="us-west-2"))

    assert isinstance(adapter, UnconfiguredOcrAdapter)
    with pytest.raises(OcrUnavailableError, match=expected_hint):
        adapter.extract_text(tmp_path / "scan.png")


def test_ocr_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported OCR provider"):
        create_ocr_adapter(OcrConfig(provider="made_up_ocr"))
