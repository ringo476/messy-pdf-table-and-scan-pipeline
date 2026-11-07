"""Steps 14 and 15: check each value with ordinary code, score it, and decide what needs a person."""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from ..config import Settings
from ..models import CheckResult, ExtractedDocument, FieldValue, PageType
from .rules import detect_document_type, has_possible_item_table

REQUIRED_FIELDS = ("invoice_number", "invoice_date", "vendor_name", "grand_total", "line_items")
_MONEY_FIELDS = ("subtotal", "tax", "grand_total", "shipping", "cgst", "sgst", "igst", "deposit")
_DATE_FIELDS = ("invoice_date", "due_date")
INVOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_\-.]*$")
_LETTERS = re.compile(r"[A-Za-z]")
# A lowercase letter followed by a capital inside one word, with the word at least four letters long.
_JOINED_WORDS = re.compile(r"\b(?=\w*[a-z][A-Z])\w{4,}\b")
# Real invoice words such as Implementation (14) or Troubleshooting (15) must not be flagged, so this is 16 or more.
_LONG_WORD = re.compile(r"[A-Za-z]{16,}")
# A lowercase l beside a capital, or a letter that looks like a digit in the middle of digits.
_LOOKALIKE = re.compile(r"[A-Z]l|l[A-Z]|\d[OoIlSsB]\d|\b[0-9]+[OoIl]\b|\b[OoIl][0-9]{2,}\b")


def _add(field: FieldValue, name: str, passed: bool, detail: str = "") -> CheckResult:
    result = CheckResult(name=name, passed=passed, detail=detail)
    field.checks.append(result)
    return result


def expected_totals(
    base: Decimal,
    tax: Decimal | None,
    cgst: Decimal | None,
    sgst: Decimal | None,
    igst: Decimal | None,
    shipping: Decimal | None,
    discount: Decimal | None,
    rounding: Decimal | None = None,
    deposit: Decimal | None = None,
) -> list[Decimal]:
    """The grand totals that a correct invoice could print, given a subtotal (or the sum of its lines).

    Tax is the tax line if there is one. Otherwise it is the sum of the CGST, SGST and IGST lines. Shipping and the
    rounding adjustment (as printed, so it may be negative) are added. A discount or a deposit is taken off, or is only
    informational if the amount shown was already reduced, so each can go either way. The first total returned is
    the one with everything applied.
    """
    tax_total = tax if tax is not None else sum((part for part in (cgst, sgst, igst) if part is not None), Decimal(0))
    core = base + tax_total + (shipping or Decimal(0)) + (rounding or Decimal(0))
    discounts = [abs(discount), Decimal(0)] if discount else [Decimal(0)]
    deposits = [abs(deposit), Decimal(0)] if deposit else [Decimal(0)]
    return [core - off - paid for off in discounts for paid in deposits]


def _amounts(fields: dict[str, FieldValue]) -> dict[str, Decimal | None]:
    return {
        name: (fields[name].value if isinstance(fields[name].value, Decimal) else None)
        for name in (
            "subtotal",
            "tax",
            "grand_total",
            "discount",
            "shipping",
            "cgst",
            "sgst",
            "igst",
            "rounding",
            "deposit",
        )  # fmt: skip
    }


def _total_matches(base: Decimal, amounts: dict[str, Decimal | None], tolerance: Decimal) -> tuple[bool, list[Decimal]]:
    total = amounts["grand_total"]
    candidates = expected_totals(
        base,
        amounts["tax"],
        amounts["cgst"],
        amounts["sgst"],
        amounts["igst"],
        amounts["shipping"],
        amounts["discount"],
        amounts["rounding"],
        amounts["deposit"],
    )
    return (total is not None and any(abs(candidate - total) <= tolerance for candidate in candidates)), candidates


def _invoice_has_no_item_table_and_adds_up(
    fields: dict[str, FieldValue], document: ExtractedDocument, tolerance: Decimal
) -> bool:
    """A one-line bill has no item table. That is fine only when nothing suggests a table was missed.

    The page must have no table at all and no missed-text recovery, and the subtotal and charges must add up
    to the total. Otherwise an item table the layout model missed would make the line items vanish unnoticed.
    """
    if has_possible_item_table(document) or any(page.missed_text_check_ran for page in document.pages):
        return False
    amounts = _amounts(fields)
    subtotal = amounts["subtotal"]
    return subtotal is not None and _total_matches(subtotal, amounts, tolerance)[0]


def _check_required(fields: dict[str, FieldValue], document: ExtractedDocument, settings: Settings) -> None:
    tolerance = Decimal(str(settings.money_tolerance))
    for name in REQUIRED_FIELDS:
        field = fields[name]
        empty = field.value in (None, "", [])
        if empty and name == "line_items" and _invoice_has_no_item_table_and_adds_up(fields, document, tolerance):
            _add(field, "required_present", True, "this invoice has no item table, and its amounts add up")
            continue
        _add(field, "required_present", not empty, "this required field is empty" if empty else "")


def _check_formats(fields: dict[str, FieldValue], credit_note: bool) -> None:
    number = fields["invoice_number"]
    if number.value is not None:
        _add(
            number,
            "valid_format",
            bool(INVOICE_ID.match(str(number.value))),
            "invoice numbers use letters, digits and - / _ .",
        )
    for name in _DATE_FIELDS:
        field = fields[name]
        if field.value is None:
            if field.raw:
                _add(field, "valid_format", False, f"could not read '{field.raw}' as a date")
            continue
        day = date.fromisoformat(str(field.value))
        _add(field, "valid_format", 2000 <= day.year <= 2100, "the date's year is outside 2000 to 2100")
    for name in _MONEY_FIELDS:
        field = fields[name]
        if field.value is not None:
            # A credit note may print its amounts as negatives. An invoice never should.
            _add(
                field,
                "valid_format",
                isinstance(field.value, Decimal) and (credit_note or field.value >= 0),
                "amounts must not be negative",
            )
    vendor = fields["vendor_name"]
    if vendor.value is not None:
        _add(vendor, "valid_format", bool(_LETTERS.search(str(vendor.value))), "a vendor name must contain letters")
    invoice_date, due_date = fields["invoice_date"], fields["due_date"]
    if invoice_date.value and due_date.value:
        _add(
            due_date,
            "after_invoice_date",
            str(due_date.value) >= str(invoice_date.value),
            "the due date is before the invoice date",
        )


def _check_arithmetic(fields: dict[str, FieldValue], settings: Settings, credit_note: bool) -> list[CheckResult]:
    """Checks that each line, the subtotal, the extra charges and the grand total add up."""
    tolerance = Decimal(str(settings.money_tolerance))
    results: list[CheckResult] = []
    items_field = fields["line_items"]
    all_amounts = _amounts(fields)
    subtotal, total = all_amounts["subtotal"], all_amounts["grand_total"]
    items: list[dict[str, Any]] = items_field.value or []

    # Line arithmetic: quantity x unit price = amount. A credit note may print the sign either way.
    if items:
        wrong, untestable = [], 0
        for number, item in enumerate(items, start=1):
            quantity, price, amount = item.get("quantity"), item.get("unit_price"), item.get("amount")
            if quantity is None or price is None or amount is None:
                untestable += 1
                continue
            product = quantity * price
            difference = abs(abs(product) - abs(amount)) if credit_note else abs(product - amount)
            if difference > tolerance + Decimal("0.005"):
                wrong.append(number)
        if wrong:
            _add(
                items_field,
                "line_arithmetic",
                False,
                f"quantity x unit price doesn't equal the amount on lines {wrong[:10]}",
            )
        elif untestable:
            _add(items_field, "line_arithmetic", False, f"{untestable} lines are missing a quantity, price or amount")
        else:
            _add(items_field, "line_arithmetic", True)

    # Lines add up to the subtotal. With no subtotal, the lines plus the tax and charges must make the total.
    line_amounts: list[Decimal] = [item["amount"] for item in items if item.get("amount") is not None]
    if items and len(line_amounts) == len(items):
        items_sum = sum(line_amounts, Decimal(0))
        if subtotal is not None:
            ok = abs(items_sum - subtotal) <= tolerance
            detail = "" if ok else f"the line items add up to {items_sum}, but the subtotal is {subtotal}"
            _add(items_field, "lines_add_up", ok, detail)
            _add(fields["subtotal"], "lines_add_up", ok, detail)
        elif total is not None:
            ok, candidates = _total_matches(items_sum, all_amounts, tolerance)
            detail = (
                ""
                if ok
                else f"the line items plus tax and charges come to {candidates[0]}, but the grand total is {total}"
            )
            _add(items_field, "lines_add_up", ok, detail)
            _add(fields["grand_total"], "lines_add_up", ok, detail)

    # Subtotal + tax + charges - discount = grand total.
    if subtotal is not None and total is not None:
        ok, candidates = _total_matches(subtotal, all_amounts, tolerance)
        detail = "" if ok else f"subtotal plus tax and other charges is {candidates[0]}, but the grand total is {total}"
        for name in (
            "grand_total", "subtotal", "tax", "shipping", "discount", "cgst", "sgst", "igst", "rounding", "deposit",
        ):  # fmt: skip
            if name in ("grand_total", "subtotal") or fields[name].value is not None:
                _add(fields[name], "total_adds_up", ok, detail)
    elif total is not None and not items:
        _add(
            fields["grand_total"],
            "total_adds_up",
            False,
            "there are no line items or subtotal to check the total against",
        )
    return results


def _check_reading_quality(fields: dict[str, FieldValue], settings: Settings) -> None:
    """Applies only to OCR-read values, because text-layer words have no reading uncertainty."""
    for field in fields.values():
        if field.ocr_confidence is not None and not field.llm_confirmed:
            ok = field.ocr_confidence >= settings.min_ocr_confidence
            _add(field, "read_clearly", ok, f"OCR confidence is only {field.ocr_confidence:.2f}" if not ok else "")


def _check_lookalike_characters(fields: dict[str, FieldValue]) -> None:
    """Flags an OCR-read ID that mixes look-alike characters, such as 'Sl-123' for 'SI-123'.

    Arithmetic can't catch these, and one wrong character makes an invoice number useless. Text-layer
    values are exact, so they skip this check.
    """
    number = fields["invoice_number"]
    if number.value and number.source != "text_layer" and not number.llm_confirmed:
        suspicious = _LOOKALIKE.search(str(number.value))
        detail = f"OCR may have confused look-alike characters in '{suspicious.group(0)}'" if suspicious else ""
        _add(number, "no_lookalike_characters", suspicious is None, detail)


def _check_vendor_guess(fields: dict[str, FieldValue]) -> None:
    """A vendor guessed from the top of the page is only trusted if the rest of the page was understood.

    On an invoice in a language the label rules don't know, the first line is often the word 'Invoice' in that language,
    or a logo. If neither the invoice number nor the total was found by its label, the page wasn't understood, and the
    guess can't be trusted, however plausible it looks.
    """
    vendor = fields["vendor_name"]
    guessed_from_heading = vendor.value and vendor.evidence_penalty > 0
    if guessed_from_heading:
        understood = fields["invoice_number"].value not in (None, "") and fields["grand_total"].value is not None
        _add(
            vendor,
            "page_understood",
            understood,
            "" if understood else "the vendor was guessed from the top of a page whose labels weren't recognised",
        )


def _check_joined_words(fields: dict[str, FieldValue]) -> None:
    """Flags an OCR-read name whose words have been run together, such as 'RiversideClinic'.

    OCR sometimes drops the space between two words. Nothing else can tell, because a name has no arithmetic.
    A capital letter inside a word is also how names like McDonald are written, so those are sent to a person
    too. That costs a look, and it never lets a mangled name through. Text-layer names are exact, so they skip this.
    """
    for name in ("vendor_name", "bill_to"):
        field = fields[name]
        if field.value and field.source != "text_layer" and not field.llm_confirmed:
            joined = _JOINED_WORDS.search(str(field.value))
            detail = f"OCR may have run two words together in '{joined.group(0)}'" if joined else ""
            _add(field, "no_joined_words", joined is None, detail)

    # Item descriptions have no arithmetic to protect them either. A word of 16 or more letters is very likely two
    # words run together. Shorter pairs such as 'Officechair' look like real words, so this check can't catch them.
    items = fields["line_items"]
    if items.value and items.source in ("ocr", "mixed"):
        for number, item in enumerate(items.value, start=1):
            long_word = _LONG_WORD.search(str(item.get("description") or ""))
            if long_word:
                _add(
                    items,
                    "no_joined_words",
                    False,
                    f"OCR may have run words together in '{long_word.group(0)}' on line {number}",
                )
                break
        else:
            _add(items, "no_joined_words", True)


def _check_table_structure(fields: dict[str, FieldValue], document: ExtractedDocument) -> None:
    """Checks that every row fits the columns, and that amount columns hold numbers."""
    items_field = fields["line_items"]
    if not items_field.value:
        return
    table = next((t for t in document.tables if t.id == items_field.raw), None)
    if table is None:
        return
    uneven = [i for i, row in enumerate(table.rows) if len(row) != table.n_cols]
    unreadable = [i + 1 for i, item in enumerate(items_field.value) if item.get("amount") is None]
    _add(
        items_field,
        "table_structure",
        not uneven and not unreadable,
        f"rows that don't fit: {uneven[:5]}; unreadable amounts on lines: {unreadable[:5]}"
        if uneven or unreadable
        else "",
    )


def _check_ambiguity(fields: dict[str, FieldValue]) -> None:
    for field in fields.values():
        if field.suggested_value is not None:
            _add(
                field,
                "second_reader_agrees",
                False,
                f"The language model read this as '{field.suggested_value}', so a person needs to confirm it",
            )
        if field.alternatives:
            _add(
                field,
                "single_value",
                False,
                f"more than one value was found: {[str(v) for v in field.alternatives[:3]]}",
            )
        if field.value is not None or field.raw:
            for note in field.rule_notes:
                _add(field, "unambiguous", False, note)


def run_checks(fields: dict[str, FieldValue], document: ExtractedDocument, settings: Settings) -> None:
    """Runs every check. Re-running after a value changes starts from a clean slate."""
    credit_note = detect_document_type(document) == "credit_note"
    for field in fields.values():
        # Checks that need the PDF or earlier invoices are worked out once, outside this module, and kept here.
        field.checks = list(field.extra_checks)
    _check_required(fields, document, settings)
    _check_formats(fields, credit_note)
    _check_arithmetic(fields, settings, credit_note)
    _check_reading_quality(fields, settings)
    _check_lookalike_characters(fields)
    _check_joined_words(fields)
    _check_vendor_guess(fields)
    _check_table_structure(fields, document)
    _check_ambiguity(fields)


def score_and_decide(fields: dict[str, FieldValue], document: ExtractedDocument, settings: Settings) -> None:
    """Step 15. A field is accepted only if all its checks pass and its score meets the minimum.

    The score starts from how the words were read. A confirming sum raises it. A failed check caps
    it below any usable minimum, so a high score can never hide a failed check.
    """
    page_types = {page.number: page.type for page in document.pages}
    for name, field in fields.items():
        if field.value in (None, "", []) and not field.raw:
            # A required field is only a problem while its required_present check fails. A one-line bill with
            # no item table passes that check, so its empty line items need no person.
            required = name in REQUIRED_FIELDS and any(
                check.name == "required_present" and not check.passed for check in field.checks
            )
            field.confidence = 0.0
            failed_reasons = [check.detail or check.name for check in field.checks if not check.passed]
            field.needs_review = required or bool(failed_reasons)
            field.review_reasons = failed_reasons
            continue

        if field.source == "text_layer":
            base = 0.97
        elif field.source == "llm":
            base = 0.9
        elif field.ocr_confidence is not None:
            # OCR base: a strong reading gets near 0.9 before any confirmation.
            base = 0.9 * field.ocr_confidence
        else:
            base = 0.8
        if field.page is not None and page_types.get(field.page) is PageType.SCANNED_BAD_TEXT:
            base -= 0.02
        base -= field.evidence_penalty
        if field.llm_confirmed:
            base = max(base, 0.9)

        arithmetic_passed = any(
            check.passed and check.name in ("lines_add_up", "total_adds_up") for check in field.checks
        )
        if arithmetic_passed:
            base = min(0.995, base + 0.08)

        failed = [check for check in field.checks if not check.passed]
        field.confidence = round(min(base, 0.5) if failed else base, 3)
        reasons = [check.detail or check.name for check in failed]
        if not failed and field.confidence < settings.min_field_score:
            reasons.append(f"confidence {field.confidence:.2f} is below the minimum of {settings.min_field_score:.2f}")
        field.needs_review = bool(reasons)
        field.review_reasons = reasons
