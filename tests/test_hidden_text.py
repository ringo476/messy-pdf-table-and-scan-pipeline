"""A digital page's text layer can say one thing while the page shows another. These tests run the real models."""

import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from pdfpipe.config import Settings
from pdfpipe.engines import Engines
from pdfpipe.fields.visible import compare_with_page
from pdfpipe.models import FieldValue
from pdfpipe.process import process_pdf

pytestmark = pytest.mark.models

HEADER = [
    ("Vendor: Acme Office Supply", 700),
    ("Invoice No: INV-1042", 670),
    ("Invoice Date: 05 Mar 2026", 650),
    ("Subtotal 100.00", 400),
    ("Tax 10.00", 380),
]


def text_as_picture(text: str) -> ImageReader:
    """The text drawn into a picture, so it is visible on the page but is not in the text layer."""
    buffer = io.BytesIO()
    page = canvas.Canvas(buffer, pagesize=(260, 24))
    page.setFont("Helvetica", 13)
    page.drawString(4, 7, text)
    page.save()
    rendered = pdfium.PdfDocument(buffer.getvalue())[0].render(scale=3).to_pil().convert("RGB")
    png = io.BytesIO()
    rendered.save(png, "PNG")
    png.seek(0)
    return ImageReader(png)


def make_pdf(path: Path, how: str) -> None:
    page = canvas.Canvas(str(path))
    page.setFont("Helvetica", 12)
    for text, y in HEADER:
        page.drawString(60, y, text)
    if how == "honest":
        page.drawString(60, 360, "Grand Total 110.00")
    elif how == "white_text":
        page.setFillColorRGB(1, 1, 1)  # Invisible on a white page.
        page.drawString(60, 360, "Grand Total 110.00")
    elif how == "picture_differs":
        # The page shows 500.00, but the text layer says 400.00.
        page.drawImage(text_as_picture("Grand Total 500.00"), 56, 355, width=260, height=24)
        hidden = page.beginText(60, 360)
        hidden.setTextRenderMode(3)
        hidden.setFont("Helvetica", 12)
        hidden.textLine("Grand Total 400.00")
        page.drawText(hidden)
    page.save()


def total_check(result_fields: dict[str, FieldValue], name: str = "grand_total") -> tuple[bool, str] | None:
    for check in result_fields[name].checks:
        if check.name == "matches_visible_text":
            return check.passed, check.detail
    return None


def test_an_honest_page_passes(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    make_pdf(tmp_path / "honest.pdf", "honest")
    result = process_pdf(tmp_path / "honest.pdf", settings, engines, tmp_path)
    checked = {name: total_check(result.fields, name) for name in ("grand_total", "invoice_number", "subtotal", "tax")}
    assert all(outcome is not None and outcome[0] for outcome in checked.values()), checked


def test_white_text_is_caught(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    make_pdf(tmp_path / "white.pdf", "white_text")
    result = process_pdf(tmp_path / "white.pdf", settings, engines, tmp_path)
    outcome = total_check(result.fields)
    assert outcome is not None
    assert outcome[0] is False
    assert "nothing is visible" in outcome[1]
    assert result.fields["grand_total"].needs_review is True


def test_a_page_that_shows_a_different_number_is_caught(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    make_pdf(tmp_path / "differs.pdf", "picture_differs")
    result = process_pdf(tmp_path / "differs.pdf", settings, engines, tmp_path)
    assert str(result.fields["grand_total"].value) == "400.00"
    outcome = total_check(result.fields)
    assert outcome is not None
    assert outcome[0] is False
    assert result.fields["grand_total"].needs_review is True
    assert result.status == "needs_review"


@pytest.mark.parametrize(
    ("written", "seen", "expected"),
    [
        ("110.00", "Grand Total 110.00", True),
        ("110.00", "110,00", True),  # a different decimal mark is still the same digits
        ("INV-1042", "INV-lO42", True),  # OCR look-alikes are tolerated
        ("400.00", "500.00", False),
        ("1,400.00", "1,500.00", False),
        ("11400.00", "11500.00", False),
        ("110.00", "", False),
    ],
)
def test_comparison_rules(written: str, seen: str, expected: bool) -> None:
    assert compare_with_page(written, seen)[0] is expected
