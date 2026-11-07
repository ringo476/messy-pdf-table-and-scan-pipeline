"""End-to-end tests with the real local models, on generated documents."""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from pdfpipe.config import Settings
from pdfpipe.engines import Engines
from pdfpipe.evaluation.synth import make_case
from pdfpipe.models import PageType, RegionType
from pdfpipe.process import process_pdf, result_to_dict

pytestmark = pytest.mark.models


def test_blank_image_gives_no_words(engines: Engines) -> None:
    assert engines.read_words(np.full((200, 400, 3), 255, dtype=np.uint8)) == []


@pytest.mark.parametrize(
    ("condition", "page_type"),
    [
        ("digital", PageType.DIGITAL),
        ("scan_light", PageType.SCANNED),
        ("bad_text_layer", PageType.SCANNED_BAD_TEXT),
        ("mixed", PageType.MIXED),
    ],
)
def test_classifies_and_reads_each_page_type(
    condition: str, page_type: PageType, tmp_path: Path, settings: Settings, engines: Engines
) -> None:
    case = make_case("ruled", condition, 11, tmp_path)
    result = process_pdf(case.pdf, settings, engines, tmp_path)
    assert result.pages[0].type is page_type
    assert result.fields["grand_total"].value == Decimal(str(case.truth["fields"]["grand_total"]))


def test_sideways_page_is_turned_upright(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    case = make_case("ruled", "sideways", 11, tmp_path)
    result = process_pdf(case.pdf, settings, engines, tmp_path)
    assert result.pages[0].rotated_degrees in (90, 270)
    assert result.fields["grand_total"].value == Decimal(str(case.truth["fields"]["grand_total"]))


def test_chart_is_saved_and_never_extracted(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    case = make_case("with_chart", "digital", 11, tmp_path)
    result = process_pdf(case.pdf, settings, engines, tmp_path)
    charts = [region for page in result.pages for region in page.regions if region.type is RegionType.CHART]
    assert charts
    assert charts[0].extracted is False
    assert charts[0].image is not None
    assert (tmp_path / charts[0].image).exists()


def test_table_across_pages_is_joined(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    case = make_case("long", "digital", 11, tmp_path)
    result = process_pdf(case.pdf, settings, engines, tmp_path)
    assert result.page_count >= 2
    assert any(table.joined for table in result.tables)
    assert len(result.fields["line_items"].value) == len(case.truth["line_items"])


@pytest.mark.parametrize("condition", ["wrong_total", "missing_invoice_number"])
def test_bad_documents_are_never_accepted(condition: str, tmp_path: Path, settings: Settings, engines: Engines) -> None:
    case = make_case("ruled", condition, 11, tmp_path)
    assert process_pdf(case.pdf, settings, engines, tmp_path).status == "needs_review"


def test_instructions_inside_a_document_change_nothing(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    case = make_case("ruled", "injection", 11, tmp_path)
    result = process_pdf(case.pdf, settings, engines, tmp_path)
    assert result.fields["grand_total"].value == Decimal(str(case.truth["fields"]["grand_total"]))


def test_result_is_plain_json_with_exact_amounts(tmp_path: Path, settings: Settings, engines: Engines) -> None:
    import json

    case = make_case("ruled", "digital", 11, tmp_path)
    data = result_to_dict(process_pdf(case.pdf, settings, engines, tmp_path))
    text = json.dumps(data)
    assert isinstance(data["fields"]["grand_total"]["value"], str)
    assert data["pages"][0]["type"] == "digital"
    assert "words" not in data["pages"][0]["regions"][0]
    assert "tables" in data and text
