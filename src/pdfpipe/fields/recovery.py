"""A second pass over fields that failed a check on a scanned page.

OCR can misread a digit in one try and get it right in another, when the same line is read zoomed in, sharpened, or in
pure black and white. For each field that failed a check, this step tries those other ways of reading just that line.

It never decides what is correct. A new reading replaces the old one only if every check passes with it, and it
doesn't make any other field worse:

- A money value needs the arithmetic to confirm it. A wrong digit that makes the totals add up is very unlikely.
- An invoice number or a date has no arithmetic to confirm it, so at least two different ways of reading must agree.

A field that no other reading fixes stays as it was and goes to a person. Reading the line again can't create trust the
checks don't give.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from ..config import Settings
from ..engines import Engines
from ..models import ExtractedDocument, FieldValue, Page, PageType, Region, RegionType
from ..pipeline import imaging
from . import checks, rules

logger = logging.getLogger(__name__)

# A field can only be re-read this way if it came from OCR and failed one of the checks about the reading itself.
READING_CHECKS = {"read_clearly", "no_lookalike_characters", "total_adds_up", "lines_add_up", "valid_format"}
ARITHMETIC_CHECKS = {"total_adds_up", "lines_add_up"}
MONEY_FIELDS = ("subtotal", "tax", "grand_total", "shipping", "discount", "cgst", "sgst", "igst")
NEEDS_AGREEMENT = ("invoice_number", "invoice_date", "due_date")
CHECKED_FIELDS = (*MONEY_FIELDS, *NEEDS_AGREEMENT)
_MAX_SIDE = 1900  # The OCR model shrinks anything larger, which would undo the zoom.
_LEFT, _RIGHT, _UP_DOWN = 450, 100, 18  # How much of the line around the value is re-read, in pixels.


def _zoom(image: np.ndarray, factor: float) -> np.ndarray:
    factor = min(factor, _MAX_SIDE / max(image.shape[:2]))
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _zoomed(image: np.ndarray) -> np.ndarray:
    return _zoom(image, 2.0)


def _sharpened(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(_zoom(image, 3.0), cv2.COLOR_BGR2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(grey)
    blurred = cv2.GaussianBlur(grey, (0, 0), 2.0)
    return cv2.cvtColor(cv2.addWeighted(grey, 1.8, blurred, -0.8, 0), cv2.COLOR_GRAY2BGR)


def _black_and_white(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(_zoom(image, 2.0), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


STRATEGIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "zoomed": _zoomed,
    "sharpened": _sharpened,
    "black and white": _black_and_white,
}


def _band(image: np.ndarray, field: FieldValue) -> np.ndarray | None:
    """The line the value sits on, with the label to its left."""
    if field.position is None:
        return None
    height, width = image.shape[:2]
    x0, y0, x1, y1 = field.position
    left, top = max(0, int(x0) - _LEFT), max(0, int(y0) - _UP_DOWN)
    right, bottom = min(width, int(x1) + _RIGHT), min(height, int(y1) + _UP_DOWN)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _read_again(image: np.ndarray, name: str, field: FieldValue, strategy: str, engines: Engines) -> FieldValue | None:
    """Reads the field's line another way, and finds the value again with the same rules as the first time."""
    band = _band(image, field)
    if band is None:
        return None
    treated = STRATEGIES[strategy](band)
    framed = cv2.copyMakeBorder(treated, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    words = engines.read_words(framed)
    if not words:
        return None
    height, width = framed.shape[:2]
    region = Region(type=RegionType.PARAGRAPH, bbox=(0, 0, width, height), score=1.0, words=words)
    document = ExtractedDocument(
        pages=[Page(number=field.page or 1, type=PageType.SCANNED, width=width, height=height, regions=[region])]
    )
    if name == "invoice_number":
        found = rules.find_invoice_number(document)
    elif name in ("invoice_date", "due_date"):
        found = rules.find_date(document, name)
    else:
        found = rules.find_money(document, name)
    if found.value is None:
        return None
    # It is the same field in the same place. Only the reading is new.
    found.page, found.position = field.page, field.position
    return found


def _render_page(pdf_path: Path, page: Page, settings: Settings, engines: Engines) -> np.ndarray | None:
    """The page as the pipeline saw it when it found the value, so the saved positions still line up."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        pdf_page = document[page.number - 1]
        try:
            scan = page.type in (PageType.SCANNED, PageType.SCANNED_BAD_TEXT)
            image, _ = imaging.render_page(pdf_page, settings.scan_dpi if scan else settings.digital_dpi, settings)
        finally:
            pdf_page.close()
    finally:
        document.close()
    if page.type in (PageType.SCANNED, PageType.SCANNED_BAD_TEXT):
        image = imaging.clean_scan(image, engines).image
    # If the image doesn't match, the saved positions can't be trusted on it.
    return image if image.shape[:2] == (page.height, page.width) else None


def _failing(fields: dict[str, FieldValue]) -> set[str]:
    return {name for name, field in fields.items() if field.needs_review}


def _tries(fields: dict[str, FieldValue]) -> list[str]:
    """Fields worth a second reading: read by OCR, and failed a check about the reading."""
    names = []
    for name in CHECKED_FIELDS:
        field = fields[name]
        failed = {check.name for check in field.checks if not check.passed}
        # A value that couldn't be parsed has no value, but its raw text and position are there to read again.
        unread = field.value is None and not field.raw
        if unread or field.position is None or field.page is None or field.source not in ("ocr", "mixed"):
            continue
        if field.needs_review and failed & READING_CHECKS:
            names.append(name)
    return names


def _passes_everywhere(
    fields: dict[str, FieldValue], name: str, candidate: FieldValue, document: ExtractedDocument, settings: Settings
) -> bool:
    """Whether every check passes with this reading, and no other field is worse off."""
    trial = {key: field.model_copy(deep=True) for key, field in fields.items()}
    trial[name] = candidate
    checks.run_checks(trial, document, settings)
    checks.score_and_decide(trial, document, settings)
    return not trial[name].needs_review and _failing(trial) <= _failing(fields)


def _enough(name: str, field: FieldValue, value: str, readings: list[tuple[str, FieldValue]]) -> bool:
    """Whether the readings that found this value are enough to trust it.

    A different money value that makes the arithmetic work is confirmed by the arithmetic, so one reading does it.
    Anything else needs two different ways of reading to agree.
    """
    confirmed_by_arithmetic = name not in NEEDS_AGREEMENT and value != str(field.value)
    return len(readings) >= (1 if confirmed_by_arithmetic else 2)


def second_pass(
    fields: dict[str, FieldValue],
    document: ExtractedDocument,
    pdf_path: Path,
    settings: Settings,
    engines: Engines,
    render: Callable[[Page], np.ndarray | None] | None = None,
) -> list[str]:
    """Tries other readings for failed fields. Returns the names of the fields that were fixed.

    render is only for tests. It gives the page image, so the real PDF isn't needed.
    """
    started = time.monotonic()
    pages = {page.number: page for page in document.pages}
    images: dict[int, np.ndarray | None] = {}
    fixed: list[str] = []

    for name in _tries(fields)[: settings.recovery_max_fields]:
        if time.monotonic() - started > settings.recovery_max_seconds:
            logger.info("the second pass ran out of time")
            break
        field = fields[name]
        assert field.page is not None
        page = pages.get(field.page)
        if page is None:
            continue
        if field.page not in images:
            images[field.page] = (render or (lambda p: _render_page(pdf_path, p, settings, engines)))(page)
        image = images[field.page]
        if image is None:
            continue

        # Every reading that passes all the checks, grouped by the value it found.
        passing: dict[str, list[tuple[str, FieldValue]]] = {}
        for strategy in STRATEGIES:
            try:
                candidate = _read_again(image, name, field, strategy, engines)
            except Exception:  # A reading that fails is the same as a reading that finds nothing.
                logger.warning("the %s reading of %s failed", strategy, name, exc_info=True)
                continue
            if candidate is None or not _passes_everywhere(fields, name, candidate, document, settings):
                continue
            passing.setdefault(str(candidate.value), []).append((strategy, candidate))
            # A different money value that makes the arithmetic work is confirmed by the arithmetic.
            if name not in NEEDS_AGREEMENT and str(candidate.value) != str(field.value):
                break

        winners = [readings for value, readings in passing.items() if _enough(name, field, value, readings)]
        if len(winners) != 1:  # No reading was trusted, or different trusted readings disagree.
            continue
        best = max((candidate for _, candidate in winners[0]), key=lambda candidate: candidate.ocr_confidence or 0.0)
        best.recovery = f"read again ({', '.join(strategy for strategy, _ in winners[0])}), and every check passed"
        fields[name] = best
        checks.run_checks(fields, document, settings)
        checks.score_and_decide(fields, document, settings)
        fixed.append(name)
        logger.info("the second pass fixed %s", name)
    return fixed
