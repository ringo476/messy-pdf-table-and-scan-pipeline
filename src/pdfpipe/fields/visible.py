"""Checks that a value read from a PDF's text layer is also what the page shows.

A digital page has two versions of its text: the text layer, which the pipeline reads, and the picture a person
sees. They can differ. A PDF can print 500 and hide 400 in the text layer, or hide text in white. Nothing else in
the pipeline compares them, because a digital page is never run through OCR.

For the few fields that matter most, this step renders the page, reads only the small area around each value with
OCR, and compares. A value that OCR can't see, or sees differently, fails a check and goes to a person.
"""

import logging
from pathlib import Path

import pypdfium2 as pdfium

from ..config import Settings
from ..engines import Engines
from ..models import CheckResult, FieldValue, PageType
from ..pipeline import imaging, words

logger = logging.getLogger(__name__)

# The values where a hidden difference does the most harm.
CHECKED_FIELDS = ("invoice_number", "invoice_date", "due_date", "subtotal", "tax", "grand_total")
# OCR mixes up these pairs, so both sides are reduced to one symbol before comparing. 4 and 5 stay different.
_LOOKALIKES = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2"})


def _squash(text: str) -> str:
    """Lowercase letters and digits only, with look-alike characters merged."""
    return "".join(character for character in text.lower() if character.isalnum()).translate(_LOOKALIKES)


def compare_with_page(written: str, seen: str) -> tuple[bool, str]:
    """Whether the text OCR saw contains what the text layer says. Returns the result and a reason."""
    if not _squash(seen):
        return False, f"'{written}' is in the PDF's text layer, but nothing is visible there on the page"
    if _squash(written) in _squash(seen):
        return True, ""
    return False, f"the text layer says '{written}', but the page shows '{seen.strip()}'"


def verify_visible_values(
    fields: dict[str, FieldValue], pdf_path: Path, page_types: dict[int, PageType], settings: Settings, engines: Engines
) -> None:
    """Adds a matches_visible_text check to each checked field that came from a digital or mixed page's text layer."""
    targets: dict[int, list[str]] = {}
    for name in CHECKED_FIELDS:
        field = fields[name]
        if field.value in (None, "") or field.source != "text_layer" or field.page is None or field.position is None:
            continue
        if page_types.get(field.page) in (PageType.DIGITAL, PageType.MIXED):
            targets.setdefault(field.page, []).append(name)
    if not targets:
        return

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for number, names in targets.items():
            page = document[number - 1]
            try:
                # Rendered like the pipeline renders a digital page, so positions line up pixel for pixel.
                image, _ = imaging.render_page(page, settings.digital_dpi, settings)
            finally:
                page.close()
            for name in names:
                field = fields[name]
                assert field.position is not None
                try:
                    seen = " ".join(word.text for word in words.ocr_box(image, field.position, engines))
                    ok, detail = compare_with_page(field.raw or str(field.value), seen)
                except Exception:  # Failing to check is never a reason to skip the check.
                    logger.exception("could not compare %s with the page", name)
                    ok, detail = False, "the value could not be compared with what the page shows"
                field.extra_checks.append(CheckResult(name="matches_visible_text", passed=ok, detail=detail))
    finally:
        document.close()
