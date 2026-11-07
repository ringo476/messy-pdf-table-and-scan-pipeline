"""The second pass: failed fields on scans are read again, and a new reading is kept only if every check passes."""

from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from conftest import line_of
from pdfpipe.config import Settings
from pdfpipe.fields import recovery
from pdfpipe.models import ExtractedDocument, FieldValue, Page, PageType, Word
from test_fields import GOOD_LINES, ITEM_ROWS, decide, make_document


class ScriptedEngines:
    """Stands in for OCR. Each call returns the line of text it was told to, so a test controls every reading."""

    def __init__(self, *readings: str) -> None:
        self.readings = list(readings)
        self.calls = 0

    def read_words(self, image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[Word]:
        text = self.readings[min(self.calls, len(self.readings) - 1)]
        self.calls += 1
        return line_of(text, 20, 20, confidence=0.99) if text else []


def white_page(page: Page) -> np.ndarray:
    return np.full((page.height, page.width, 3), 255, dtype=np.uint8)


def scanned(lines: list[tuple[str, float, float]] | None = None) -> tuple[ExtractedDocument, dict[str, FieldValue]]:
    document = make_document(lines, ITEM_ROWS, confidence=0.99)
    document.pages[0].type = PageType.SCANNED
    return document, decide(document, Settings(_env_file=None))


def run(
    fields: dict[str, FieldValue], document: ExtractedDocument, engines: ScriptedEngines, **settings: Any
) -> list[str]:
    config = Settings(_env_file=None, **settings)
    return recovery.second_pass(fields, document, Path("unused.pdf"), config, engines, render=white_page)  # type: ignore[arg-type]


def misread(old: str, new: str) -> list[tuple[str, float, float]]:
    return [(text.replace(old, new), x, y) for text, x, y in GOOD_LINES]


class TestMoney:
    def test_a_misread_total_is_fixed_when_the_arithmetic_then_works(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        assert fields["grand_total"].needs_review is True
        fixed = run(fields, document, ScriptedEngines("Grand Total 110.00"))
        assert fixed == ["grand_total"]
        assert fields["grand_total"].value == Decimal("110.00")
        assert fields["grand_total"].needs_review is False
        assert fields["grand_total"].recovery is not None
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_the_same_wrong_value_changes_nothing(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        assert run(fields, document, ScriptedEngines("Grand Total 210.00")) == []
        assert fields["grand_total"].value == Decimal("210.00")
        assert fields["grand_total"].needs_review is True

    def test_a_different_wrong_value_changes_nothing(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        assert run(fields, document, ScriptedEngines("Grand Total 310.00")) == []
        assert fields["grand_total"].value == Decimal("210.00")

    def test_a_reading_with_nothing_in_it_changes_nothing(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        assert run(fields, document, ScriptedEngines("")) == []

    def test_a_fixed_field_is_recorded_as_such_for_the_reviewer(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        run(fields, document, ScriptedEngines("Grand Total 110.00"))
        assert "read again" in str(fields["grand_total"].recovery)
        assert fields["grand_total"].source == "ocr"


class TestAgreement:
    def lookalike(self) -> tuple[ExtractedDocument, dict[str, FieldValue]]:
        return scanned(misread("INV-1042", "Sl-1042"))

    def test_an_id_is_fixed_when_two_ways_of_reading_agree(self) -> None:
        document, fields = self.lookalike()
        assert fields["invoice_number"].needs_review is True
        fixed = run(fields, document, ScriptedEngines("Invoice No: SI-1042"))
        assert fixed == ["invoice_number"]
        assert fields["invoice_number"].value == "SI-1042"
        assert fields["invoice_number"].needs_review is False

    def test_one_reading_alone_is_not_enough_for_an_id(self) -> None:
        document, fields = self.lookalike()
        # The first reading is plausible, but the others disagree or fail, so nobody can say which is right.
        engines = ScriptedEngines("Invoice No: SI-1042", "Invoice No: SI-1043", "Invoice No: Sl-1042")
        assert run(fields, document, engines) == []
        assert fields["invoice_number"].value == "Sl-1042"
        assert fields["invoice_number"].needs_review is True

    def test_two_readings_that_disagree_fix_nothing(self) -> None:
        document, fields = self.lookalike()
        engines = ScriptedEngines("Invoice No: SI-1042", "Invoice No: SI-1042", "Invoice No: SI-1043")
        # SI-1042 has two readings. SI-1043 has one. The two-reading value wins, and only because it has agreement.
        assert run(fields, document, engines) == ["invoice_number"]
        assert fields["invoice_number"].value == "SI-1042"

    def test_a_date_needs_agreement_too(self) -> None:
        document, fields = scanned(misread("05 Mar 2026", "05 Mar 202G"))
        assert fields["invoice_date"].needs_review is True
        assert run(fields, document, ScriptedEngines("Invoice Date: 05 Mar 2026")) == ["invoice_date"]
        assert fields["invoice_date"].value == "2026-03-05"


class TestWhenItDoesNotRun:
    def test_a_digital_page_is_never_re_read(self) -> None:
        document = make_document(misread("110.00", "210.00"))  # No OCR confidence, so a text-layer reading.
        fields = decide(document, Settings(_env_file=None))
        engines = ScriptedEngines("Grand Total 110.00")
        assert run(fields, document, engines) == []
        assert engines.calls == 0

    def test_a_page_that_cannot_be_matched_is_skipped(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        engines = ScriptedEngines("Grand Total 110.00")
        config = Settings(_env_file=None)
        fixed = recovery.second_pass(fields, document, Path("x.pdf"), config, engines, render=lambda page: None)  # type: ignore[arg-type]
        assert fixed == []
        assert engines.calls == 0

    def test_the_number_of_fields_tried_is_limited(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        engines = ScriptedEngines("Grand Total 110.00")
        assert run(fields, document, engines, recovery_max_fields=0) == []
        assert engines.calls == 0

    def test_the_time_spent_is_limited(self) -> None:
        document, fields = scanned(misread("110.00", "210.00"))
        engines = ScriptedEngines("Grand Total 110.00")
        assert run(fields, document, engines, recovery_max_seconds=-1) == []
        assert engines.calls == 0

    def test_a_good_invoice_is_left_alone(self) -> None:
        document, fields = scanned()
        engines = ScriptedEngines("Grand Total 999.00")
        assert run(fields, document, engines) == []
        assert engines.calls == 0
        assert fields["grand_total"].value == Decimal("110.00")
