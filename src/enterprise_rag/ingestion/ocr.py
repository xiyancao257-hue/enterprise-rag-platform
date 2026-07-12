from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class OcrUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrResult:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class OcrAdapter(Protocol):
    def extract_text(self, path: Path) -> OcrResult:
        """Extract text from an image or scanned document."""


class PdfPageRenderer(Protocol):
    def render_pages(self, path: Path, output_dir: Path) -> tuple[Path, ...]:
        """Render PDF pages to image files and return them in page order."""


class DisabledOcrAdapter:
    def extract_text(self, path: Path) -> OcrResult:
        raise OcrUnavailableError(f"OCR is not configured for {path}.")


@dataclass(frozen=True)
class TesseractOcrAdapter:
    command: str = "tesseract"
    timeout_seconds: float = 30.0

    def extract_text(self, path: Path) -> OcrResult:
        if path.suffix.lower() == ".pdf":
            raise OcrUnavailableError(
                "Tesseract image OCR does not read PDFs directly. Convert PDF pages to images first "
                "or configure a document OCR provider."
            )

        try:
            result = subprocess.run(
                [self.command, str(path), "stdout"],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise OcrUnavailableError(f"Tesseract command `{self.command}` was not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrUnavailableError(
                f"Tesseract OCR timed out after {self.timeout_seconds:.1f} seconds for {path}."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip() or "no stderr"
            raise OcrUnavailableError(f"Tesseract OCR failed for {path}: {stderr}")

        return OcrResult(
            text=result.stdout.strip(),
            metadata={
                "ocr_provider": "tesseract",
                "ocr_command": self.command,
            },
        )


@dataclass(frozen=True)
class PopplerPdfPageRenderer:
    command: str = "pdftoppm"
    dpi: int = 200
    timeout_seconds: float = 60.0

    def render_pages(self, path: Path, output_dir: Path) -> tuple[Path, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = output_dir / "page"
        try:
            result = subprocess.run(
                [self.command, "-png", "-r", str(self.dpi), str(path), str(output_prefix)],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise OcrUnavailableError(f"PDF renderer command `{self.command}` was not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrUnavailableError(
                f"PDF rendering timed out after {self.timeout_seconds:.1f} seconds for {path}."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip() or "no stderr"
            raise OcrUnavailableError(f"PDF rendering failed for {path}: {stderr}")

        pages = tuple(sorted(output_dir.glob("page-*.png"), key=_rendered_page_number))
        if not pages:
            raise OcrUnavailableError(f"PDF renderer produced no page images for {path}.")
        return pages


def _rendered_page_number(path: Path) -> int:
    suffix = path.stem.rsplit("-", maxsplit=1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return 0


@dataclass(frozen=True)
class UnconfiguredOcrAdapter:
    provider: str
    setup_hint: str

    def extract_text(self, path: Path) -> OcrResult:
        raise OcrUnavailableError(
            f"OCR provider `{self.provider}` is selected, but no runtime adapter is configured for {path}. "
            f"{self.setup_hint}"
        )
