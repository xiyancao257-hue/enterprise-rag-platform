import subprocess
from pathlib import Path

import pytest

from enterprise_rag.config import OcrConfig
from enterprise_rag.ingestion.ocr import (
    DisabledOcrAdapter,
    OcrUnavailableError,
    PopplerPdfPageRenderer,
    TesseractOcrAdapter,
    UnconfiguredOcrAdapter,
)
from enterprise_rag.ingestion.ocr_factory import create_ocr_adapter, create_pdf_page_renderer


def test_disabled_ocr_adapter_raises_clear_error(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"

    with pytest.raises(OcrUnavailableError, match="OCR is not configured"):
        DisabledOcrAdapter().extract_text(image)


def test_ocr_factory_returns_disabled_adapter_by_default() -> None:
    adapter = create_ocr_adapter(OcrConfig())

    assert isinstance(adapter, DisabledOcrAdapter)


def test_ocr_factory_returns_tesseract_adapter() -> None:
    adapter = create_ocr_adapter(
        OcrConfig(
            provider="tesseract",
            tesseract_cmd="/opt/bin/tesseract",
            tesseract_timeout_seconds=12.5,
        )
    )

    assert isinstance(adapter, TesseractOcrAdapter)
    assert adapter.command == "/opt/bin/tesseract"
    assert adapter.timeout_seconds == 12.5


def test_pdf_renderer_factory_returns_poppler_renderer() -> None:
    renderer = create_pdf_page_renderer(
        OcrConfig(
            pdf_renderer_cmd="/opt/bin/pdftoppm",
            pdf_render_dpi=300,
            pdf_render_timeout_seconds=45.0,
        )
    )

    assert isinstance(renderer, PopplerPdfPageRenderer)
    assert renderer.command == "/opt/bin/pdftoppm"
    assert renderer.dpi == 300
    assert renderer.timeout_seconds == 45.0


@pytest.mark.parametrize(
    ("provider", "expected_hint"),
    [
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


def test_tesseract_adapter_runs_cli_and_returns_ocr_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake image bytes")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=" OCR text \n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = TesseractOcrAdapter(command="tesseract-test", timeout_seconds=3).extract_text(image)

    assert result.text == "OCR text"
    assert result.metadata == {
        "ocr_provider": "tesseract",
        "ocr_command": "tesseract-test",
    }
    assert calls[0][0][0] == ["tesseract-test", str(image), "stdout"]
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is False
    assert calls[0][1]["text"] is True
    assert calls[0][1]["timeout"] == 3


def test_tesseract_adapter_reports_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "scan.png"

    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OcrUnavailableError, match="was not found"):
        TesseractOcrAdapter(command="missing-tesseract").extract_text(image)


def test_tesseract_adapter_reports_failed_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "scan.png"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="bad image")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OcrUnavailableError, match="bad image"):
        TesseractOcrAdapter().extract_text(image)


def test_tesseract_adapter_rejects_pdf_without_page_rendering(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"

    with pytest.raises(OcrUnavailableError, match="does not read PDFs directly"):
        TesseractOcrAdapter().extract_text(pdf)


def test_poppler_pdf_renderer_runs_cli_and_returns_pages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "scan.pdf"
    output_dir = tmp_path / "pages"
    pdf.write_bytes(b"fake pdf")

    def fake_run(*args, **kwargs):
        output_dir.joinpath("page-10.png").write_bytes(b"10")
        output_dir.joinpath("page-2.png").write_bytes(b"2")
        output_dir.joinpath("page-1.png").write_bytes(b"1")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    pages = PopplerPdfPageRenderer(command="pdftoppm-test", dpi=250, timeout_seconds=9).render_pages(pdf, output_dir)

    assert [page.name for page in pages] == ["page-1.png", "page-2.png", "page-10.png"]


def test_poppler_pdf_renderer_reports_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "scan.pdf"

    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OcrUnavailableError, match="was not found"):
        PopplerPdfPageRenderer(command="missing-pdftoppm").render_pages(pdf, tmp_path / "pages")


def test_poppler_pdf_renderer_reports_failed_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "scan.pdf"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="bad pdf")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OcrUnavailableError, match="bad pdf"):
        PopplerPdfPageRenderer().render_pages(pdf, tmp_path / "pages")
