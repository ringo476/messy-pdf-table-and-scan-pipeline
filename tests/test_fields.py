"""Tests for steps 13 to 15: the rules, the checks and the decision."""

from decimal import Decimal

from conftest import document_of
from pdfpipe.config import Settings
from pdfpipe.fields import checks, rules
from pdfpipe.models import ExtractedDocument, FieldValue, JoinedTable, Page, PageType, Region, RegionType

GOOD_LINES = [
    ("Vendor: Acme Office Supply", 100, 50),
    ("Invoice No: INV-1042", 100, 120),
    ("Invoice Date: 05 Mar 2026", 100, 160),
    ("Due Date: 2026-04-05", 100, 200),
    ("Bill To: Delta Finance", 100, 240),
    ("Currency: USD", 100, 280),
    ("Subtotal 100.00", 700, 900),
    ("Tax 10.00", 700, 940),
    ("Grand Total 110.00", 700, 980),
]
ITEM_ROWS = [
    ["Description", "Qty", "Unit Price", "Amount"],
    ["Pens", "2", "20.00", "40.00"],
    ["Paper", "3", "20.00", "60.00"],
]


def make_document(
    lines: list[tuple[str, float, float]] | None = None,
    rows: list[list[str]] | None = None,
    confidence: float | None = None,
) -> ExtractedDocument:
    document = document_of(lines if lines is not None else GOOD_LINES, confidence)
    item_rows = rows if rows is not None else ITEM_ROWS
    document.tables = [
        JoinedTable(
            id="table_1",
            pages=[1],
            n_cols=len(item_rows[0]),
            rows=item_rows,
            row_sources=[(1, (100, 400 + i * 30, 900, 430 + i * 30)) for i in range(len(item_rows))],
            row_min_confidences=[confidence for _ in item_rows],
        )
    ]
    return document


def decide(document: ExtractedDocument, settings: Settings) -> dict[str, FieldValue]:
    fields = rules.extract_fields(document)
    checks.run_checks(fields, document, settings)
    checks.score_and_decide(fields, document, settings)
    return fields


def failed_checks(field: FieldValue) -> list[str]:
    return [check.name for check in field.checks if not check.passed]


class TestRules:
    def test_finds_every_field(self, settings: Settings) -> None:
        fields = decide(make_document(), settings)
        assert fields["invoice_number"].value == "INV-1042"
        assert fields["invoice_date"].value == "2026-03-05"
        assert fields["due_date"].value == "2026-04-05"
        assert fields["bill_to"].value == "Delta Finance"
        assert fields["currency"].value == "USD"
        assert fields["subtotal"].value == Decimal("100.00")
        assert fields["tax"].value == Decimal("10.00")
        assert fields["grand_total"].value == Decimal("110.00")
        assert len(fields["line_items"].value) == 2

    def test_a_clean_invoice_is_accepted(self, settings: Settings) -> None:
        fields = decide(make_document(), settings)
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_every_value_keeps_its_page_and_position(self, settings: Settings) -> None:
        field = decide(make_document(), settings)["invoice_number"]
        assert field.page == 1
        assert field.position is not None
        assert field.source == "text_layer"

    def test_due_date_is_not_mistaken_for_the_invoice_date(self, settings: Settings) -> None:
        lines = [("Date: 2026-01-02", 100, 100), ("Due Date: 2026-02-02", 100, 140)]
        fields = rules.extract_fields(document_of(lines))
        assert fields["invoice_date"].value == "2026-01-02"
        assert fields["invoice_date"].alternatives == []
        assert fields["due_date"].value == "2026-02-02"

    def test_subtotal_is_not_mistaken_for_the_total(self) -> None:
        lines = [("Subtotal 90.00", 100, 100), ("Total 99.00", 100, 140)]
        fields = rules.extract_fields(document_of(lines))
        assert fields["grand_total"].value == Decimal("99.00")
        assert fields["subtotal"].value == Decimal("90.00")

    def test_grand_total_beats_a_plain_total(self) -> None:
        lines = [("Total 90.00", 100, 100), ("Grand Total 99.00", 100, 140)]
        assert rules.extract_fields(document_of(lines))["grand_total"].value == Decimal("99.00")

    def test_two_labels_on_one_line(self) -> None:
        lines = [("Invoice No: A-77 Date: 2026-01-02", 100, 100)]
        fields = rules.extract_fields(document_of(lines))
        assert fields["invoice_number"].value == "A-77"
        assert fields["invoice_date"].value == "2026-01-02"

    def test_value_under_its_label(self) -> None:
        lines = [("Invoice No", 100, 100), ("B-55", 100, 128)]
        assert rules.extract_fields(document_of(lines))["invoice_number"].value == "B-55"

    def test_ocr_text_with_no_space_after_the_label(self) -> None:
        lines = [("Invoice No:C-99", 100, 100), ("Bill To:Delta Finance", 100, 140)]
        fields = rules.extract_fields(document_of(lines, confidence=0.99))
        assert fields["invoice_number"].value == "C-99"
        assert fields["bill_to"].value == "Delta Finance"

    def test_a_sentence_mentioning_a_label_is_not_a_field(self) -> None:
        lines = [("Payment is due by the date shown on this invoice for all items", 100, 100)]
        assert rules.extract_fields(document_of(lines))["invoice_date"].value is None

    def test_missing_field_stays_empty(self) -> None:
        fields = rules.extract_fields(document_of([("Hello world", 100, 100)]))
        assert fields["invoice_number"].value is None
        assert fields["grand_total"].value is None

    def test_summary_rows_are_not_line_items(self, settings: Settings) -> None:
        rows = [*ITEM_ROWS, ["Subtotal", "", "", "100.00"], ["Total", "", "", "110.00"]]
        fields = decide(make_document(rows=rows), settings)
        assert len(fields["line_items"].value) == 2

    def test_wrapped_description_joins_the_item_above(self, settings: Settings) -> None:
        rows = [ITEM_ROWS[0], ["Pens blue", "2", "20.00", "40.00"], ["fine point", "", "", ""], ITEM_ROWS[2]]
        items = decide(make_document(rows=rows), settings)["line_items"].value
        assert len(items) == 2
        assert items[0]["description"] == "Pens blue fine point"

    def test_two_row_header_is_understood(self, settings: Settings) -> None:
        rows = [["Description", "Qty", "Price", ""], ["", "", "Unit Price", "Amount"], *ITEM_ROWS[1:]]
        items = decide(make_document(rows=rows), settings)["line_items"].value
        assert [item["amount"] for item in items] == [Decimal("40.00"), Decimal("60.00")]

    def test_a_table_without_item_headers_is_ignored(self, settings: Settings) -> None:
        rows = [["Name", "Phone"], ["Sam", "555"]]
        assert decide(make_document(rows=rows), settings)["line_items"].value is None


def without(label: str, lines: list[tuple[str, float, float]] | None = None) -> list[tuple[str, float, float]]:
    return [line for line in (lines if lines is not None else GOOD_LINES) if label not in line[0]]


class TestExtraCharges:
    """Totals that include tax lines, shipping, discounts and split GST must still be checked, not given up on."""

    def test_a_missing_subtotal_no_longer_leaves_the_total_unchecked(self, settings: Settings) -> None:
        lines = without("Subtotal")
        good = decide(make_document(lines), settings)
        assert "lines_add_up" in [check.name for check in good["grand_total"].checks if check.passed]
        assert good["grand_total"].needs_review is False

    def test_a_wrong_total_is_caught_when_the_subtotal_is_missing(self, settings: Settings) -> None:
        lines = [(text.replace("110.00", "210.00"), x, y) for text, x, y in without("Subtotal")]
        fields = decide(make_document(lines), settings)
        assert "lines_add_up" in failed_checks(fields["grand_total"])
        assert fields["grand_total"].needs_review is True

    def totals_block(self, *rows: str) -> list[tuple[str, float, float]]:
        """The header lines of a good invoice, then the given total rows, one visual line each."""
        header = [line for line in GOOD_LINES if line[1] < 700]
        return [*header, *[(text, 700, 900 + 40 * index) for index, text in enumerate(rows)]]

    def test_shipping_and_a_discount_are_part_of_the_sum(self, settings: Settings) -> None:
        lines = self.totals_block(
            "Subtotal 100.00", "Tax 10.00", "Shipping 15.00", "Discount 5.00", "Grand Total 120.00"
        )
        fields = decide(make_document(lines), settings)
        assert fields["shipping"].value == Decimal("15.00")
        assert fields["discount"].value == Decimal("5.00")
        assert fields["grand_total"].needs_review is False

    def test_a_discount_that_is_only_informational_also_adds_up(self, settings: Settings) -> None:
        # The subtotal was already reduced, so the total is subtotal plus tax and shipping.
        lines = self.totals_block(
            "Subtotal 100.00", "Tax 10.00", "Shipping 15.00", "Discount 5.00", "Grand Total 125.00"
        )
        assert decide(make_document(lines), settings)["grand_total"].needs_review is False

    def test_a_total_that_matches_neither_reading_goes_to_review(self, settings: Settings) -> None:
        lines = self.totals_block(
            "Subtotal 100.00", "Tax 10.00", "Shipping 15.00", "Discount 5.00", "Grand Total 130.00"
        )
        fields = decide(make_document(lines), settings)
        assert "total_adds_up" in failed_checks(fields["grand_total"])

    def test_split_gst_lines_add_up_with_no_tax_line(self, settings: Settings) -> None:
        lines = self.totals_block("Subtotal 100.00", "CGST 9% 9.00", "SGST 9% 9.00", "Grand Total 118.00")
        fields = decide(make_document(lines), settings)
        assert fields["cgst"].value == Decimal("9.00")
        assert fields["sgst"].value == Decimal("9.00")
        assert fields["tax"].value is None
        assert fields["grand_total"].needs_review is False

    def test_split_gst_that_does_not_add_up_is_caught(self, settings: Settings) -> None:
        lines = self.totals_block("Subtotal 100.00", "CGST 9% 9.00", "SGST 9% 9.00", "Grand Total 128.00")
        assert decide(make_document(lines), settings)["grand_total"].needs_review is True

    def test_a_tax_rate_in_the_label_is_not_the_value(self) -> None:
        lines = [("VAT (20%) 20.00", 700, 100), ("IGST @ 18% 18.00", 700, 140)]
        fields = rules.extract_fields(document_of(lines))
        assert fields["tax"].value == Decimal("20.00")
        assert fields["igst"].value == Decimal("18.00")

    def test_shipping_and_discount_rows_are_not_line_items(self, settings: Settings) -> None:
        rows = [*ITEM_ROWS, ["Shipping and handling", "", "", "15.00"], ["Rabatt", "", "", "-5.00"]]
        assert len(decide(make_document(rows=rows), settings)["line_items"].value) == 2


class TestInvoicesWithoutAnItemTable:
    def one_line_bill(self, total: str = "Grand Total 110.00") -> ExtractedDocument:
        lines = [*without("Grand Total"), (total, 700, 980)]
        document = document_of(lines)
        document.tables = []
        return document

    def test_a_one_line_bill_that_adds_up_needs_no_person(self, settings: Settings) -> None:
        fields = decide(self.one_line_bill(), settings)
        assert fields["line_items"].needs_review is False
        assert fields["line_items"].value is None
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_it_still_goes_to_review_when_the_amounts_do_not_add_up(self, settings: Settings) -> None:
        fields = decide(self.one_line_bill("Grand Total 210.00"), settings)
        assert fields["line_items"].needs_review is True
        assert fields["grand_total"].needs_review is True

    def test_it_goes_to_review_if_the_missed_text_check_had_to_recover_words(self, settings: Settings) -> None:
        # A table the layout model missed would look the same, so a page that needed recovery is not trusted.
        document = self.one_line_bill()
        document.pages[0].missed_text_check_ran = True
        assert decide(document, settings)["line_items"].needs_review is True

    def test_a_table_that_could_not_be_read_as_items_still_goes_to_review(self, settings: Settings) -> None:
        document = make_document(rows=[["Name", "Phone"], ["Sam", "555"]])
        assert decide(document, settings)["line_items"].needs_review is True

    def test_without_a_subtotal_a_missing_table_is_never_accepted(self, settings: Settings) -> None:
        document = document_of(without("Subtotal"))
        document.tables = []
        assert decide(document, settings)["line_items"].needs_review is True


class TestTotalsBoxAndJoinedWords:
    def test_a_totals_box_is_not_an_item_table(self, settings: Settings) -> None:
        # Layout models often detect the small Subtotal / Tax / Total box as a table of its own.
        lines = [*without("Grand Total"), ("Grand Total 110.00", 700, 980)]
        document = document_of(lines)
        document.tables = [
            JoinedTable(
                id="table_1",
                pages=[1],
                n_cols=2,
                rows=[["Subtotal", "100.00"], ["Tax", "10.00"], ["Grand Total", "110.00"]],
            )
        ]
        fields = decide(document, settings)
        assert fields["line_items"].needs_review is False
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_a_table_that_is_not_a_totals_box_could_hide_the_items(self, settings: Settings) -> None:
        document = document_of(GOOD_LINES)
        document.tables = [
            JoinedTable(
                id="table_1", pages=[1], n_cols=2, rows=[["Part", "Cost"], ["Pens", "40.00"], ["Paper", "60.00"]]
            )
        ]
        assert decide(document, settings)["line_items"].needs_review is True

    def test_an_ocr_name_with_two_words_run_together_goes_to_review(self, settings: Settings) -> None:
        lines = [(text.replace("Delta Finance", "DeltaFinance"), x, y) for text, x, y in GOOD_LINES]
        field = decide(make_document(lines, confidence=0.99), settings)["bill_to"]
        assert "no_joined_words" in failed_checks(field)

    def test_the_same_name_from_the_text_layer_is_exact(self, settings: Settings) -> None:
        lines = [(text.replace("Delta Finance", "DeltaFinance"), x, y) for text, x, y in GOOD_LINES]
        assert decide(make_document(lines), settings)["bill_to"].needs_review is False

    def test_an_ocr_item_description_with_words_run_together_goes_to_review(self, settings: Settings) -> None:
        # This was a real silent error: every number was right, but the description had lost a space.
        rows = [ITEM_ROWS[0], ["Whiteboardmarkers", "2", "20.00", "40.00"], ITEM_ROWS[2]]
        items = decide(make_document(rows=rows, confidence=0.99), settings)["line_items"]
        assert "no_joined_words" in failed_checks(items)
        assert "line 1" in items.review_reasons[0]

    def test_the_same_description_from_the_text_layer_is_exact(self, settings: Settings) -> None:
        rows = [ITEM_ROWS[0], ["Whiteboardmarkers", "2", "20.00", "40.00"], ITEM_ROWS[2]]
        assert decide(make_document(rows=rows), settings)["line_items"].needs_review is False

    def test_ordinary_long_words_are_fine(self, settings: Settings) -> None:
        rows = [ITEM_ROWS[0], ["Implementation and troubleshooting", "2", "20.00", "40.00"], ITEM_ROWS[2]]
        assert decide(make_document(rows=rows, confidence=0.99), settings)["line_items"].needs_review is False

    def test_a_short_pair_run_together_cannot_be_detected(self, settings: Settings) -> None:
        # The limit of this check, written down so nobody assumes more than it does.
        rows = [ITEM_ROWS[0], ["Officechair", "2", "20.00", "40.00"], ITEM_ROWS[2]]
        assert decide(make_document(rows=rows, confidence=0.99), settings)["line_items"].needs_review is False

    def test_an_ordinary_ocr_name_is_not_flagged(self, settings: Settings) -> None:
        assert decide(make_document(confidence=0.99), settings)["bill_to"].needs_review is False


class TestCreditNotes:
    LINES = [
        ("Credit Note", 100, 20),
        *[(text, x, y) for text, x, y in GOOD_LINES[:6]],
        ("Subtotal -100.00", 700, 900),
        ("Tax -10.00", 700, 940),
        ("Grand Total -110.00", 700, 980),
    ]
    ROWS = [ITEM_ROWS[0], ["Pens", "2", "20.00", "-40.00"], ["Paper", "3", "20.00", "-60.00"]]

    def test_the_type_is_detected(self) -> None:
        assert rules.detect_document_type(make_document(self.LINES, self.ROWS)) == "credit_note"
        assert rules.detect_document_type(make_document()) == "invoice"

    def test_negative_amounts_on_a_credit_note_are_accepted(self, settings: Settings) -> None:
        fields = decide(make_document(self.LINES, self.ROWS), settings)
        assert fields["grand_total"].value == Decimal("-110.00")
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_a_credit_note_whose_total_is_wrong_is_still_caught(self, settings: Settings) -> None:
        lines = [(text.replace("-110.00", "-210.00"), x, y) for text, x, y in self.LINES]
        assert decide(make_document(lines, self.ROWS), settings)["grand_total"].needs_review is True

    def test_negative_amounts_on_an_invoice_are_still_refused(self, settings: Settings) -> None:
        lines = [(text.replace("Tax 10.00", "Tax -10.00"), x, y) for text, x, y in GOOD_LINES]
        assert "valid_format" in failed_checks(decide(make_document(lines), settings)["tax"])


class TestOtherLanguages:
    def test_a_german_invoice(self, settings: Settings) -> None:
        lines = [
            ("Lieferant: Müller Bürobedarf GmbH", 100, 50),
            ("Rechnungsnummer: RE-2026-77", 100, 120),
            ("Rechnungsdatum: 15. März 2026", 100, 160),
            ("Fälligkeitsdatum: 15.04.2026", 100, 200),
            ("Rechnung an: Delta Finanz", 100, 240),
            ("Zwischensumme 100,00", 700, 900),
            ("MwSt 19% 19,00", 700, 940),
            ("Gesamtbetrag 119,00", 700, 980),
        ]
        rows = [
            ["Beschreibung", "Menge", "Einzelpreis", "Betrag"],
            ["Stifte", "2", "20,00", "40,00"],
            ["Papier", "3", "20,00", "60,00"],
        ]
        fields = decide(make_document(lines, rows), settings)
        assert fields["invoice_number"].value == "RE-2026-77"
        assert fields["invoice_date"].value == "2026-03-15"
        assert fields["due_date"].value == "2026-04-15"
        assert fields["subtotal"].value == Decimal("100.00")
        assert fields["tax"].value == Decimal("19.00")
        assert fields["grand_total"].value == Decimal("119.00")
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_a_french_invoice(self, settings: Settings) -> None:
        lines = [
            ("Fournisseur: Papeterie Martin SARL", 100, 50),
            ("Facture N° F-2026-5", 100, 120),
            ("Date de facture: 5 mars 2026", 100, 160),
            ("Date d'échéance: 5 avril 2026", 100, 200),
            ("Client: Delta Finance", 100, 240),
            ("Total HT 100,00", 700, 900),
            ("TVA 20% 20,00", 700, 940),
            ("Total TTC 120,00", 700, 980),
        ]
        rows = [
            ["Désignation", "Quantité", "Prix unitaire", "Montant"],
            ["Stylos", "2", "20,00", "40,00"],
            ["Papier", "3", "20,00", "60,00"],
        ]
        fields = decide(make_document(lines, rows), settings)
        assert fields["invoice_number"].value == "F-2026-5"
        assert fields["invoice_date"].value == "2026-03-05"
        assert fields["grand_total"].value == Decimal("120.00")
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_a_spanish_invoice(self, settings: Settings) -> None:
        lines = [
            ("Proveedor: Papelería Sol SL", 100, 50),
            ("Factura Nº: A-55", 100, 120),
            ("Fecha de factura: 5 de marzo de 2026", 100, 160),
            ("Fecha de vencimiento: 5 de abril de 2026", 100, 200),
            ("Cliente: Delta Finanzas", 100, 240),
            ("Base imponible 100,00", 700, 900),
            ("IVA 21% 21,00", 700, 940),
            ("Total a pagar 121,00", 700, 980),
        ]
        rows = [
            ["Descripción", "Cantidad", "Precio unitario", "Importe"],
            ["Bolígrafos", "2", "20,00", "40,00"],
            ["Papel", "3", "20,00", "60,00"],
        ]
        fields = decide(make_document(lines, rows), settings)
        assert fields["invoice_number"].value == "A-55"
        assert fields["invoice_date"].value == "2026-03-05"
        assert fields["grand_total"].value == Decimal("121.00")
        assert [name for name, field in fields.items() if field.needs_review] == []

    def test_an_unknown_label_is_left_empty_not_guessed(self) -> None:
        fields = rules.extract_fields(document_of([("Fakturanummer: X-1", 100, 100)]))
        assert fields["invoice_number"].value is None


class TestVendorFallback:
    def test_a_detail_line_is_not_taken_as_the_vendor(self) -> None:
        document = document_of([("Reference: see attached", 100, 50), ("Acme Office Supply", 100, 90)])
        assert rules.extract_fields(document)["vendor_name"].value == "Acme Office Supply"


class TestChecks:
    def replace(self, old: str, new: str) -> list[tuple[str, float, float]]:
        return [(text.replace(old, new), x, y) for text, x, y in GOOD_LINES]

    def test_wrong_grand_total_fails_and_needs_review(self, settings: Settings) -> None:
        fields = decide(make_document(self.replace("110.00", "210.00")), settings)
        assert "total_adds_up" in failed_checks(fields["grand_total"])
        assert fields["grand_total"].needs_review is True
        assert fields["grand_total"].confidence <= 0.5

    def test_line_items_not_adding_up_fails(self, settings: Settings) -> None:
        fields = decide(make_document(self.replace("Subtotal 100.00", "Subtotal 150.00")), settings)
        assert "lines_add_up" in failed_checks(fields["line_items"])
        assert fields["subtotal"].needs_review is True

    def test_wrong_line_arithmetic_fails(self, settings: Settings) -> None:
        rows = [ITEM_ROWS[0], ["Pens", "2", "20.00", "45.00"], ["Paper", "3", "20.00", "55.00"]]
        fields = decide(make_document(rows=rows), settings)
        assert "line_arithmetic" in failed_checks(fields["line_items"])

    def test_missing_required_field_needs_review(self, settings: Settings) -> None:
        lines = [line for line in GOOD_LINES if "Invoice No" not in line[0]]
        assert decide(make_document(lines), settings)["invoice_number"].needs_review is True

    def test_missing_optional_field_does_not_need_review(self, settings: Settings) -> None:
        lines = [line for line in GOOD_LINES if "Due Date" not in line[0]]
        assert decide(make_document(lines), settings)["due_date"].needs_review is False

    def test_ambiguous_date_needs_review(self, settings: Settings) -> None:
        field = decide(make_document(self.replace("05 Mar 2026", "03/04/2026")), settings)["invoice_date"]
        assert field.needs_review is True

    def test_unreadable_date_needs_review(self, settings: Settings) -> None:
        field = decide(make_document(self.replace("05 Mar 2026", "16 Jul J 2026")), settings)["invoice_date"]
        assert field.needs_review is True

    def test_two_different_values_need_review(self, settings: Settings) -> None:
        lines = [*GOOD_LINES, ("Invoice No: INV-9999", 100, 1200)]
        field = decide(make_document(lines), settings)["invoice_number"]
        assert "single_value" in failed_checks(field)

    def test_due_date_before_invoice_date_needs_review(self, settings: Settings) -> None:
        field = decide(make_document(self.replace("2026-04-05", "2020-01-01")), settings)["due_date"]
        assert field.needs_review is True

    def test_low_ocr_confidence_needs_review(self, settings: Settings) -> None:
        field = decide(make_document(confidence=0.6), settings)["bill_to"]
        assert "read_clearly" in failed_checks(field)

    def test_ocr_lookalike_in_an_id_needs_review(self, settings: Settings) -> None:
        lines = [(text.replace("INV-1042", "Sl-1042"), x, y) for text, x, y in GOOD_LINES]
        field = decide(make_document(lines, confidence=0.99), settings)["invoice_number"]
        assert "no_lookalike_characters" in failed_checks(field)

    def test_exact_text_layer_id_skips_the_lookalike_check(self, settings: Settings) -> None:
        lines = [(text.replace("INV-1042", "Sl-1042"), x, y) for text, x, y in GOOD_LINES]
        assert decide(make_document(lines), settings)["invoice_number"].needs_review is False

    def test_a_labelled_vendor_is_accepted(self, settings: Settings) -> None:
        field = decide(make_document(), settings)["vendor_name"]
        assert field.value == "Acme Office Supply"
        assert field.needs_review is False

    def test_a_vendor_from_a_plain_line_needs_review(self, settings: Settings) -> None:
        lines = [("Acme Office Supply", 100, 50), *GOOD_LINES[1:]]
        field = decide(make_document(lines), settings)["vendor_name"]
        assert field.value == "Acme Office Supply"
        assert field.needs_review is True

    def test_a_vendor_from_a_top_heading_is_accepted_at_a_lower_score(self, settings: Settings) -> None:
        document = make_document([("Acme Office Supply", 100, 50), *GOOD_LINES[1:]])
        page = document.pages[0]
        heading_words = [w for w in page.regions[0].words if w.bbox[1] == 50]
        page.regions[0].words = [w for w in page.regions[0].words if w.bbox[1] != 50]
        page.regions.append(Region(type=RegionType.HEADING, bbox=(90, 40, 400, 80), score=0.9, words=heading_words))
        field = decide(document, settings)["vendor_name"]
        assert field.needs_review is False
        assert field.confidence < 0.97

    def test_invoice_word_is_not_taken_as_the_vendor(self) -> None:
        document = document_of([("Tax Invoice", 100, 50), ("Acme Ltd", 100, 90)])
        assert rules.extract_fields(document)["vendor_name"].value == "Acme Ltd"


class TestScoring:
    def test_a_failed_check_caps_the_score(self, settings: Settings) -> None:
        document = ExtractedDocument(pages=[Page(number=1, type=PageType.DIGITAL, width=100, height=100)])
        field = FieldValue(value="x", raw="x", source="text_layer", page=1)
        checks._add(field, "something", False, "it failed")
        fields = {"bill_to": field}
        checks.score_and_decide(fields, document, settings)
        assert field.confidence <= 0.5
        assert field.needs_review is True

    def test_low_score_without_a_failed_check_still_needs_review(self, settings: Settings) -> None:
        document = ExtractedDocument(pages=[Page(number=1, type=PageType.SCANNED, width=100, height=100)])
        field = FieldValue(value="x", raw="x", source="ocr", ocr_confidence=0.82, page=1)
        checks.score_and_decide({"bill_to": field}, document, settings)
        assert field.needs_review is True

    def test_confirming_arithmetic_raises_the_score(self, settings: Settings) -> None:
        fields = decide(make_document(confidence=0.95), settings)
        assert fields["grand_total"].confidence > fields["bill_to"].confidence
