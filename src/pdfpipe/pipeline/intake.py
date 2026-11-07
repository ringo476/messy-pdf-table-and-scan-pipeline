"""Step 1: accept the PDF only if it is safe and valid to process."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from ..config import Settings


class RejectedFile(ValueError):
    """The file can't be processed. The message is safe to show to the person who uploaded it."""


@dataclass
class IntakeInfo:
    fingerprint: str
    page_count: int
    size_bytes: int


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path, settings: Settings) -> IntakeInfo:
    """Checks the file without trusting its name or extension.

    Scripts, forms and attachments inside the PDF are never run, because the pipeline only reads
    its text and renders its pages. The limits below stop oversized files from exhausting the PC.
    """
    size = path.stat().st_size
    if size == 0:
        raise RejectedFile("The file is empty.")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise RejectedFile(f"The file is larger than the {settings.max_upload_mb} MB limit.")
    with path.open("rb") as handle:
        if b"%PDF-" not in handle.read(1024):
            raise RejectedFile("The file is not a PDF.")

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as error:
        if "password" in str(error).lower():
            raise RejectedFile("The PDF is password protected. Remove the password and upload it again.") from error
        raise RejectedFile("The PDF is damaged and can't be opened.") from error

    try:
        page_count = len(document)
        if page_count == 0:
            raise RejectedFile("The PDF has no pages.")
        if page_count > settings.max_pages:
            raise RejectedFile(f"The PDF has {page_count} pages. The limit is {settings.max_pages}.")
        for index in range(page_count):
            page = document[index]
            try:
                width, height = page.get_size()
            finally:
                page.close()
            if width <= 0 or height <= 0 or max(width, height) > settings.max_page_side_points:
                raise RejectedFile(f"Page {index + 1} has an unsupported size.")
    finally:
        document.close()

    return IntakeInfo(fingerprint=fingerprint_file(path), page_count=page_count, size_bytes=size)
